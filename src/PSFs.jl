module PSFs
using FourierTools: center_pos, FourierJoin
using FourierTools, NDTools, IndexFunArrays, SpecialFunctions, FFTW
using ZernikePolynomials
export PSFParams, sinc_r, jinc_r_2d, pupil_xyz, apsf, psf, k0, kxy, aplanatic_factor
export get_Abbe_limit, get_Nyquist_limit
export kz_mid_pos

export ModeWidefield, ModeConfocal, Mode4Pi, ModeISM, Mode2Photon
export MethodRichardsWolf, MethodPropagate, MethodPropagateIterative, MethodShell, MethodSincR

include("aplanatic.jl")
include("PSF_types.jl")
include("util.jl")
include("pupil_pol.jl")
include("pupil.jl")
include("aPSFs.jl")

"""
    psf(::Type{ModeWidefield}, sz::NTuple, pp::PSFParams; sampling=get_sampling(sz, pp))

calculates the widefield single-frequency point spread function (psf), i.e. the image of a single (very small) emitter. Most of the parameters
(such as refractive index, numerical aperture, vacuum wavelength, aberrations etc.) are hidden in the parameter structure argument `pp`,
which should be generated via the `PSFParams()` constructor. See ``PSFParams()` for details.

#Parameters
+ `sz`:         size tuple of the final PSF
+ `pp`:         PSF parameters of the PSF. See `PSFParams()` for details. The argument `pp.aplanatic` defined whether an excitation or emission PSF is calculated.
+ `sampling=nothing`:   The sampling parameters of the resulting PSF. If `nothing` is provided, the PSF will be sampled according to the Abbe limit.
+ `use_resampling=true`: Exploits a calculation trick, which first calculates the amplitudes are calculated on a twice coarser grid and the result is upsampled. This increases the speed but may be less accurate.
+ `return_amp=false`:    Has to be `false` since confocal amplitude spread functions do not exist for non-zero pinhole sizes. 

See also:
+ apsf():  calculates the underlying amplitude point spread function (apsf)

Example:
```jdoctest
julia> pp = PSFParams(0.5,1.4,1.52); sz=(128,128,128); sampling=(0.050,0.050,0.200);
# an emission PSF of an isotropic (freely rotating emitter)
julia> p_wf = psf(sz, pp; sampling=sampling);
# an emission PSF of an emitter oriented along the Z direction
julia> pp_dipole = PSFParams(0.5,1.4,1.52; transition_dipole=[0.0,0.0,1.0]);
julia> p_dipole = psf(sz, pp_dipole; sampling=sampling);

```
"""
function psf(::Type{ModeWidefield}, sz::NTuple, pp::PSFParams; sampling=nothing, use_resampling=true, return_amp=false) # unclear why the resampling seems to be so bad
    amp = let
        if use_resampling
            fct_ex = (sz,my_sampling) -> apsf(sz, pp, sampling=my_sampling, center_kz=true)
            calc_with_resampling(fct_ex, sz, sampling, norm_amp=true) # , shift_kz=kz_mid_pos(sz, pp, sampling)
        else
            apsf(sz, pp, sampling=sampling)
        end
    end
    if return_amp
        return amp_to_int(amp, pp), amp
    else
        return amp_to_int(amp, pp)
    end
end

"""
    psf(::Type{ModeConfocal}, sz::NTuple, pp_em::PSFParams; pp_ex=nothing, pinhole=nothing, pinhole_ft=disc_pinhole_ft, sampling=nothing, use_resampling=true, return_amp=false, pinhole_positions=[(0.0,0.0)]) # unclear why the resampling seems to be so bad
    Calculates a confocal point spread function. The normalisation is such that a completely open `pinhole` diameter yields the excitation PSF with its normalization. 
    Returns the PSF or a vector of PSFs.
    
#Parameters
+ `sz`:         size tuple of the final PSF
+ `pp_em`:      PSF parameters of the emission PSF. PSF parameters of the PSF. See `PSFParams()` for details. This should include the emission wavelength
+ `pp_ex=nothing`:      This is a required named parameter, containing all the settings for the excitation PSF. This should include the excitation wavelength as well as typically `aplanatic=aplanatic_illumination`.
+ `pinhole=nothing`:    The diameter of the pinhole in Airy Units (AU = 1.22 λ/NA). A pinhole size of one AU corresponds to a pinhole border falling onto the first zero of a corresponding paraxial emission PSF.
+ `sampling=nothing`:   The sampling parameters of the resulting PSF.
+ `use_resampling=true`: Exploits a calculation trick, which first calculates the individual PSFs on a twice coarser grid in XY and Z and then upsamples the result. Note that this may be inappropriate due to undersampling due to the Stokes shift which is neglected here. But warnings will then result during the calculations of the subsampled widefield PSFs.
+ `return_amp=false`:    Has to be `false` since confocal amplitude spread functions do not exist for non-zero pinhole sizes. 
+ `pinhole_positions=[(0.0,0.0)]`:  A list of pinhole positions. One PSF will be returned for each pinhole position. If only a single pinhole position is supplied the PSF will directly be returned instead of a vector of PSFs.
+ `pinhole_ft=disc_pinhole_ft`:  Specifies which function is used to calculate the Fourier transform of the pinhole. This allows the user to control the pinhole shape. 
+ `pinhole_positions=[(0.0,0.0)]`:  Specifies the precise position(s) of the pinholes in the detection path. This allows to simulate an offset (misadjusted) pinhole, or (as a vector of tuples) a PSF for a whole set of positions. See the `psf(ModeISM, ...)` for more details. 
+ `ex2p`:   If `true`, the excitation PSF is locally squared to calculate a two-photon confocal PSF. However, you should typically use the `Mode2Photon` to do this. Note that the order of the PSFParams are reversed.

```jdoctest
julia> pp_em = PSFParams(0.5,1.4,1.52; mode=ModeConfocal);
julia> pp_ex = PSFParams(pp_em; λ=0.488, aplanatic=aplanatic_illumination);
julia> p_conf = psf((128,128,128),pp_em; pp_ex=pp_ex, pinhole=0.1, sampling=(0.040,0.040,0.100));
```
"""
function psf(::Type{ModeConfocal}, sz::NTuple, pp_em::PSFParams; pp_ex=nothing, pinhole=nothing, pinhole_ft=disc_pinhole_ft, sampling=nothing, use_resampling=true, return_amp=nothing, pinhole_positions=[(0.0,0.0)], ex2p=false) # unclear why the resampling seems to be so bad
    if isnothing(pp_ex) 
        error("The named parameter `pp_ex` is obligatory for confocal calculation. Provide the excitation PSF parameters here.")
    end
    if isnothing(pinhole)
        error("The named parameter `pinhole` is obligatory for confocal calculation. Provide the excitation PSF parameters here.")
    end
    if !isnothing(return_amp) && return_amp == true
        error("A confocal PSF cannot return an amplitude. Please use `return_amp=false`.")
    end

    # creat a pseudo parameter structure with the combined wavelength just to check the individual amplitude samplings of the final result.
    λeff = 1 / (1/pp_ex.λ + 1/pp_em.λ)
    pp_both = PSFParams(pp_em; λ=λeff)
    # the factor of two below, is since the amp psf can be twice undersampled, but the intensity psf not.
    check_amp_sampling(sz, pp_both, sampling .* 2.0)

    psf_ex = let
        if use_resampling
            fct_ex = (sz,my_sampling) -> psf(ModeWidefield, sz, pp_ex; sampling=my_sampling, use_resampling=use_resampling)
            calc_with_resampling(fct_ex, sz, sampling, norm_amp=false)
        else
            psf(ModeWidefield, sz, pp_ex; sampling=sampling, use_resampling=use_resampling)
        end
    end

    # apply the two-photon effect for excitation if needed.
    psf_ex = ex2p ? abs2.(psf_ex) : psf_ex

    # pp_em = PSFParams(pp, mode = ModeWidefield)
    psf_em = let
        if use_resampling
            fct_em = (sz,my_sampling) -> psf(ModeWidefield, sz,pp_em; sampling=my_sampling, use_resampling=use_resampling)
            calc_with_resampling(fct_em, sz, sampling, norm_amp=false)
        else
            psf(ModeWidefield, sz, pp_em; sampling=sampling, use_resampling=use_resampling)
        end
    end

    # now we need to modify the sampling such that the pinhole corrsponds to the equivalent of one Airy Unit.
    # The Airy Unit is the diameter of the Airy disc: 1.22 * lamda_em / NA 
    # AU = 1.22 * pp_em.λ / pp_em.NA
    AU_pix = AU_per_pixel(pp_em, sampling) # AU ./ sampling[1:2]
    if any(pinhole .* AU_pix .> sz[1:2]) 
        @warn "Pinhole is larger than image this leads to serious aliasing artefacts. Maximal pinhole size is: $(sz[1:2]./AU_pix) AU."
    end
    # This can be done a lot more efficiently by staying in Fourier space. Ideally even by only calculating half the range of the jinc function:
    # pinhole = real.(ift2d(jinc_r_2d(sz, pinhole .* AU_pix, pp_em.dtype)))
    # pinhole_ft = rfft2d(ifftshift(pinhole))

    all_PSFs = [];

    rfft_psf_em = rfft2d(psf_em)
    for p in pinhole_positions
        my_pinhole_ft = exp_ikx_rfft(pp_em.dtype,sz, shift_by=p) .* pinhole_ft(sz, pp_em, pinhole.* AU_pix)
        # pinhole_ft = rfft2d(ifftshift(pinhole))
        # return pinhole_ft
        my_em =  irfft2d(rfft_psf_em .* my_pinhole_ft, sz[1])
        push!(all_PSFs, my_em .* psf_ex)
        # push!(all_PSFs, irfft2d(my_pinhole_ft, sz[1]))  # only for diagnostic purposes
    end

    if length(all_PSFs) == 1
        return all_PSFs[1]
    else
        return all_PSFs
    end
end

"""
    disc_pinhole_ft(sz, pp, pinhole)

calculates the Fourier transform of a disc-shaped pinhole
+ `pp`: PSF parameter structure
+ `sampling`: sampling of the image pixels in object coordinates
+ `pinhole`: diameter of the pinhole in pixels
"""
function disc_pinhole_ft(sz, pp, pinhole)
    jinc_r_2d(sz, pinhole, pp.dtype; r_func= PSFs.rr_rfft)
end

"""
    disc_pinhole_ft(sz, pp, pinhole)

calculates the Fourier transform of a box-shaped pinhole
+ `pp`: PSF parameter structure
+ `sampling`: sampling of the image pixels in object coordinates
+ `pinhole`: tuple of side lengths of the pinhole in pixels
"""
function box_pinhole_ft(sz, pp, pinhole)
    sinc_r_2d(sz, pinhole, pp.dtype)
end

"""
    AU_in_pixels(pp, sampling)

calculates the size of the Airy unites in pixels.
#arguments
+ `pp`: PSF parameter structure
+ `sampling`: sampling of the image pixels in object coordinates
"""
function AU_per_pixel(pp, sampling)
    AU = 1.22 * pp.λ / pp.NA
    AU_pix = AU ./ sampling[1:2]
    return AU_pix
end

function exp_ikx_rfft(dtype, sz; shift_by)
    szrfft = rfft_size(sz)
    ifftshift(exp_ikx(Complex{dtype}, szrfft[1:2], offset=CtrRFT, shift_by=shift_by), (2:length(sz)))
end

function ism_positions_rect(pinhole_dist, pinhole_grid)
    [pinhole_dist.*((n,m).-((pinhole_grid.+1)./2 )) for n=1:pinhole_grid[1] for m=1:pinhole_grid[2]], pinhole_dist
end

"""
    psf(::Type{ModeISM}, sz::NTuple, pp_em::PSFParams; pinhole=nothing, sampling=nothing, pinhole_ft=box_pinhole_ft, pinhole_positions=nothing, pinhole_dist=0.5, pinhole_grid=(5,5), ism_pos=ism_positions_rect, args...)
    Calculates a confocal point spread function. The normalisation is such that a completely open `pinhole` diameter yields the excitation PSF with its normalization. 
    Returns the PSF or a vector of PSFs.
    
#Parameters
+ `sz`:         size tuple of the final PSF
+ `pp_em`:      PSF parameters of the emission PSF. This should include the emission wavelength
+ `pp_ex=nothing`:      This is a required named parameter, containing all the settings for the excitation PSF. This should include the excitation wavelength as well as typically `aplanatic=aplanatic_illumination`.
+ `pinhole=nothing`:   The diameter of each pinhole in Airy Units (AU = 1.22 λ/NA). By default (pinholes=nothing) the pinhole size is automatically calculated by the `pinhole_dist` parameter below to yield mutually touching pinholes.
+ `sampling=nothing`:   The sampling parameters of the resulting PSF.
+ `use_resampling=true`: Exploits a calculation trick, which first calculates the individual PSFs on a twice coarser grid in XY and Z and then upsamples the result. Note that this may be inappropriate due toundersampling due to the Stokes shift which is neglected here. But warnings will then result during the calculations of the subsampled widefield PSFs.
+ `return_amp=false`:    Has to be `false` since confocal amplitude spread functions do not exist for non-zero pinhole sizes. 
+ `pinhole_positions=nothing`:  A list of pinhole positions. One PSF will be returned for each pinhole position. If only a single pinhole position is supplied the PSF will directly be returned instead of a vector of PSFs.
+ `pinhole_dist=0.5`: a value or tuple specified the distances between pinholes, when arranged in a grid. 
+ `pinhole_ft=box_pinhole_ft`:  Specifies which function is used to calculate the Fourier transform of the pinhole. This allows the user to control the pinhole shape. E.g. hexagonal pattern with round pinholes
+ `pinhole_positions=nothing`:  Specifies the precise positions of the pinholes in the detection pathway. Be careful, since this is not the same as the position of the relative shift of the images.

```jdoctest
julia> sz=(128,128,128); sampling = (0.04,0.04,0.120)
julia> pp_em = PSFParams(0.5,1.4,1.52; mode=ModeISM);
julia> pp_ex = PSFParams(pp_em; λ=0.488, aplanatic=aplanatic_illumination);
julia> p_ism = psf(sz,pp_em; pp_ex=pp_ex, pinhole=0.21, pinhole_dist=0.2, sampling=sampling);
```
"""
function psf(::Type{ModeISM}, sz::NTuple, pp_em::PSFParams; pinhole=nothing, sampling=nothing, pinhole_ft=box_pinhole_ft, pinhole_positions=nothing, pinhole_dist=0.5, pinhole_grid=(5,5), ism_pos=ism_positions_rect, args...)
    if isnothing(pinhole_positions) || isempty(pinhole_positions)
        #create a list of pinhole positions
        pinhole_dist = pinhole_dist .* AU_per_pixel(pp_em, sampling)
        pinhole_positions, ph_diam = ism_pos(pinhole_dist, pinhole_grid)
        # convert to AUs
        ph_diam = ph_diam ./ AU_per_pixel(pp_em, sampling)
        if !isnothing(pinhole)
            if any(pinhole .> ph_diam)
                @warn "The given pinhole size $(pinhole) AU is larger that the mutual pinhole spacing $(ph_diam). This is unphysical."
            end
        else
            pinhole = ph_diam
        end
    end
    psf(ModeConfocal, sz, pp_em; pinhole=pinhole, sampling=sampling, pinhole_ft=pinhole_ft,pinhole_positions=pinhole_positions, args...) # confocal is able to handle this well
end

"""
    psf(::Type{Mode2Photon}, sz::NTuple, pp_ex::PSFParams; pp_em=nothing, pinhole=nothing, sampling=nothing, pinhole_ft=disc_pinhole_ft, args...)
    Calculates a 2-photon (potentially confocal) point spread function.  
    Returns the PSF or a vector of PSFs.
    
#Parameters
+ `sz`:         size tuple of the final PSF
+ `pp_ex`:      PSF parameters of the excitation PSF. This should include the exission wavelength (typically in the IR region). Please make sure to also set `pp.aplanatic=aplanatic_illumination`.
+ `pp_em`:      PSF parameters of the emission PSF. This only needs to be supplied, if a pinhole is used. Otherwise non-descanned detection (NDD) is assumed.
+ `pinhole=nothing`:   If `nothing`, NDD is assumed and the PSF is only the square of the excitation PSF. The diameter of each pinhole in Airy Units (AU = 1.22 λ/NA). 
+ `sampling=nothing`:   The sampling parameters of the resulting PSF.
+ `use_resampling=true`: Exploits a calculation trick, which first calculates the individual PSFs on a twice coarser grid in XY and Z and then upsamples the result. Note that this may be inappropriate due toundersampling due to the Stokes shift which is neglected here. But warnings will then result during the calculations of the subsampled widefield PSFs.
+ `return_amp=false`:    Has to be `false` since confocal amplitude spread functions do not exist for non-zero pinhole sizes. 
+ `pinhole_positions=nothing`:  A list of pinhole positions. One PSF will be returned for each pinhole position. If only a single pinhole position is supplied the PSF will directly be returned instead of a vector of PSFs.

```jdoctest
julia> sz=(128,128,128); sampling = (0.04,0.04,0.120)
julia> pp_ex = PSFParams(0.8,1.4,1.52; mode=Mode2Photon, aplanatic= aplanatic_illumination);
julia> p_2p = psf(sz,pp_ex; sampling=sampling);
```
"""
function psf(::Type{Mode2Photon}, sz::NTuple, pp::PSFParams; pinhole=nothing, sampling=nothing, pinhole_ft=box_pinhole_ft, pinhole_positions=nothing, args...)
    if isnothing(pinhole) 
        return abs2.(psf(ModeWidefield, sz, pp; sampling=sampling))
    else
        psf(ModeConfocal, sz, pp; ex_2p=true, pinhole=pinhole, sampling=sampling, pinhole_ft=pinhole_ft,pinhole_positions=pinhole_positions, args...) # confocal is able to handle this well
    end
end


"""
    psf(sz::NTuple, pp::PSFParams; sampling=get_sampling(sz, pp))

calculates the point spread function (psf), i.e. the image of a single (very small) emitter. Most of the parameters
(such as refractive index, numerical aperture, vacuum wavelength, aberrations etc.) are hidden in the parameter structure argument `pp`,
which should be generated via the `PSFParams()` constructor. See ``PSFParams()` for details.
Note that the field `pp.mode` defines the microscopic mode to simulate. Currently implemented are the default `ModeWidefield` and `ModeConfocal`.

See also:
+ apsf():  calculates the underlying amplitude point spread function (apsf)

Example:
```jdoctest
julia> pp = PSFParams(0.5,1.4,1.52);
julia> p = psf((128,128,128),pp; sampling=(0.050,0.050,0.200));
```
"""
function psf(sz::NTuple, pp::PSFParams; nps...) # unclear why the resampling seems to be so bad
    return psf(pp.mode, sz, pp; nps...)
end

end # module
