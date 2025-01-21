"""
    pupil_θ(sz, pp::PSFParams, sampling)

returns the θ angle (to the optical axis) in the sample space as a pupil array.
"""
function pupil_θ(sz, pp::PSFParams, sampling)
    asin.(k_r(sz, pp, sampling)./k_0(pp))
end

"""
    limit_θ(theta, α_max)

limits the maximum angle to the theoretical α_max. This is particularly important for the 1/sqrt(cos θ) aplanatic factor as this factor diverges and wins over the roundoff errors of the smooth pupil.
"""
function limit_θ(theta, α_max)
    min(theta, α_max)
end
# function limit_θ(theta, pp::PSFParams)
#     min.(theta, asin(pp.dtype(pp.NA/pp.n)))
# end


"""
    pupil_ϕ(sz, pp::PSFParams, sampling)

returns the azimuthal angle ϕ  in the sample space as a pupil array.
"""
function pupil_ϕ(sz, pp::PSFParams, sampling)
    ϕ_tuple.(k_xy(sz, pp, sampling))
end

"""
    aplanatic_factor(sz, pp::PSFParams, sampling)

returns the aplanatic factor as specified in `pp.aplanatic` as a pupil array.
"""
function aplanatic_factor(sz, pp::PSFParams, sampling; is_proj=true)
    α_max = asin(pp.dtype(pp.NA/pp.n))
    if is_proj
        pp.aplanatic.(limit_θ.(pupil_θ(sz, pp, sampling),α_max))
    else
        pp.aplanatic.(limit_θ.(pupil_θ(sz, pp, sampling),α_max)) .* cos.(limit_θ.(pupil_θ(sz, pp, sampling), α_max))
    end
end

"""
    field_pupil(sz, pp, sampling)

returns the pupil polarization as a 4D array with the XY polarization components stacked along the 4th dimension.
"""
function field_pupil(sz, pp, sampling)
    idx_to_dim(expand_dims(pp.polarization.(pp.dtype, k_xy_rel_pupil(sz,pp,sampling)),Val(3)))
end

"""
    field_xy_to_xyz(field,pp,sampling)

converts a 2D field at the pupil to a 2D field behind the lens, containing E_x, E_Y and E_z.
"""
function field_xy_to_xyz(field,pp,sampling)
    if size(field)[4] == 1
        return field; # just stay scalar and apply nothing.
    end
    sz = size(field)
    ϕ = pupil_ϕ(sz, pp, sampling) # establish angles
    θ = pupil_θ(sz, pp, sampling)
    cosϕ = cos.(ϕ)
    sinϕ = sin.(ϕ)    
    cosθ = cos.(θ)
    sinθ = sin.(θ)
    field_rad = field[:,:,:,1] .* cosϕ + field[:,:,:,2] .* sinϕ # radial field component in the pupil
    field_azi = -field[:,:,:,1] .* sinϕ + field[:,:,:,2] .* cosϕ # azimuthal field component in the pupil
    field_rad_cosθ = cosθ .* field_rad # what remains of the radial field component behind lens (with energy conservation)
    cat(field_rad_cosθ .* cosϕ .- field_azi .* sinϕ, field_rad_cosθ .* sinϕ .+ field_azi .* cosϕ, sinθ .* field_rad, dims=4)
end

"""
    field_xyz(sz, pp, sampling)

creates a 2D pupil field behind the lens, containing E_x, E_Y and E_z.
"""
function field_xyz(sz, pp, sampling)
    field_xy_to_xyz(field_pupil(sz, pp, sampling), pp, sampling)
end

"""
    pupil_xyz(sz, pp, sampling=nothing)

creates a pupil with electric field distributions in XYZ. Returns a 4D dataset with the electric field components along the 4th dimension.
#Arguments
+ 'sz':     size of the pupil in pixels
+ 'pp':     the `PSFParam` structure with all the PSF parameters
+ 'sampling':   the pixel sampling in the same units as the wavelength
+ 'is_proj':    defines whether the pupil is to be interpreted as the projection of the McCutchen pupil or not. This yields a by 1 ./cos(Theta) modified aplanatic factor.
"""
function pupil_xyz(sz, pp, sampling=nothing; is_proj=true)
    if isnothing(sampling)
        sampling = get_Ewald_sampling(sz, pp)
    end
    # res = zeros(complex(pp.dtype),(sz[1],sz[2],1,3))
    if isnothing(pp.aberrations) || isempty(pp.aberrations.indices)
        field_xyz(sz, pp, sampling) .* aplanatic_factor(sz,pp,sampling, is_proj=is_proj) .* ft(jinc_r_2d(sz[1:2], pp, sampling=sampling)) #  .* my_disc(sz[1:2],pp)
    else
        field_xyz(sz, pp, sampling) .* aplanatic_factor(sz,pp,sampling, is_proj=is_proj) .* get_zernike_pupil(sz, pp, sampling) .* ft(jinc_r_2d(sz[1:2], pp, sampling=sampling) ) # .* my_disc(sz[1:2],pp)
    end
end

"""
    get_propagator(sz,pp,sampling)

retrieves the propagator phase, prpagating a single Z-slice.
"""
function get_propagator(sz, pp, sampling)
    if isnothing(sampling)
        sampling = get_Ewald_sampling(sz, pp)
    end
    k_max_rel = sampling[1:2] ./ (pp.λ / pp.n)
    # prop1 = propagator(pp.dtype, sz[1:2], Δz=1, k_max = k_max_rel)
    # prop = cispi.(2 .* phase_kz(pp.dtype, sz[1:2], scale = 1 ./ (k_max_rel .* sz[1:2])))
    scalar = (length(sampling) > 2) ? pp.dtype((2π*sampling[3] / (pp.λ / pp.n))) : 1.0
    xy_scale = pp.dtype.(1 ./ (k_max_rel .* sz[1:2]))
    return scalar .* phase_kz(pp.dtype, sz[1:2], scale = xy_scale), scalar, xy_scale
end

"""
    get_propagator_gradient(prop_phase, scalar, xy_scale)

calculates the gradient of the propagator along the X and Y directions of the pupil.
"""
function get_propagator_gradient(prop_phase, scalar, xy_scale)
    #dr_prop_phase = ifelse.(prop_phase .== zero(pp.dtype), zero(pp.dtype), scalar.^2 ./ prop_term .* rr(pp.dtype, sz[1:2], scale = xy_scale))
    dtype = eltype(prop_phase)
    sz = size(prop_phase)[1:2]
    dx_prop_phase = ifelse.(prop_phase .== zero(dtype), zero(dtype), scalar.^2 ./ prop_phase .* xx(dtype, sz, scale = xy_scale.^2))
    dy_prop_phase = ifelse.(prop_phase .== zero(dtype), zero(dtype), scalar.^2 ./ prop_phase .* yy(dtype, sz, scale = xy_scale.^2))
    return dx_prop_phase, dy_prop_phase
end

function get_Zernike_normcoeff(index_style, j)
    n,m = let 
        if index == :OSA
            OSA2mn(j)
        elseif index == :Noll
            Noll2mn(j)
        else
            error("Unknown Zernike sequential index")
        end
    end
    return normalization(Int64(n),Int64(m)) # The normalization of the Zernike toolbox is according to Thibos et al. - "Standards for Reporting the Optical Aberrations of Eyes"
end

"""
    get_zernike_pupil_phase(sz, pp, sampling) 

calculates the phases in the pupil for a given set of aberrations as defined by `J` and `coefficients`.
By default this follows the OSA nomenclature. See the help file of `ZernikePolynomials.jl` for more information.
The pupil phase (up to the pupil border as defined by the `NA` in `pp`) is returned.

Arguments:
+ `sz`:  size of the real-space array
+ `pp`:  PSF parameter structure
+ `sampling`: pixelpitch in real space as NTuple

Example:
```jdoctest
julia> using PointSpreadFunctions, FFTW

julia> aberr = PointSpreadFunctions.Aberrations([Zernike_Spherical, Zernike_ObliqueAstigmatism],[0.1, 0.2])
Aberrations([12, 3], [0.1, 0.2], :OSA)

julia> pp = PSFParams(580.0, 1.4, 1.518; aberrations=aberr)
PSFParams(580.0, 1.4, 1.518, Float32, ModeWidefield, PointSpreadFunctions.pol_scalar, PointSpreadFunctions.var"#42#43"(), PointSpreadFunctions.MethodPropagateIterative, nothing, Aberrations([12, 3], [0.1, 0.2], :OSA), nothing)

julia> sz = (10,10,64)
(10, 10, 64)

julia> sampling=(190,190,100)
(190, 190, 100)

julia> PointSpreadFunctions.get_zernike_pupil_phase(sz,pp,sampling)
10×10 Matrix{Float64}:
 0.0   0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0
 0.0   0.0         0.0         0.533599    0.202003   -0.0206203  -0.170662   -0.21173     0.0         0.0
 0.0   0.0         0.477274    0.186398    0.0287554  -0.104828   -0.250743   -0.3726     -0.361221    0.0
 0.0   0.533599    0.186398    0.0937363   0.0736565   0.016983   -0.112676   -0.278928   -0.3726     -0.21173
 0.0   0.202003    0.0287554   0.0736565   0.154747    0.162853    0.0615812  -0.112676   -0.250743   -0.170662
 0.0  -0.0206203  -0.104828    0.016983    0.162853    0.223607    0.162853    0.016983   -0.104828   -0.0206203
 0.0  -0.170662   -0.250743   -0.112676    0.0615812   0.162853    0.154747    0.0736565   0.0287554   0.202003
 0.0  -0.21173    -0.3726     -0.278928   -0.112676    0.016983    0.0736565   0.0937363   0.186398    0.533599
 0.0   0.0        -0.361221   -0.3726     -0.250743   -0.104828    0.0287554   0.186398    0.477274    0.0
 0.0   0.0         0.0        -0.21173    -0.170662   -0.0206203   0.202003    0.533599    0.0         0.0

```
"""
function get_zernike_pupil_phase(sz, pp, sampling) 
    J = pp.aberrations.indices
    if isempty(J)
        return zeros(pp.dtype, sz)
    end
    coefficients = pp.aberrations.coefficients
    index_style = get_zernike_index_style(pp.aberrations.index_style)
    border = k_pupil_pos(sz[1:2],pp,sampling[1:2])
    rho = rr(pp.dtype, sz[1:2], scale = one(pp.dtype)./border)
    rho = min.(rho, one(pp.dtype))
    phi = phiphi(pp.dtype, sz[1:2], scale = one(pp.dtype)./border)
    # X = ramp(pp.dtype, 1,sz[1],scale = 1/border[1])
    # X = min.(X, one(pp.dtype))
    # Y = ramp(pp.dtype, 1,sz[2],scale = 1/border[2])
    # Y = min.(Y, one(pp.dtype))
    # D = [[zernike(j; index=index, coord=:cartesian)(x,y) for x in X, y in Y] for j in J ]
    # lets change this to 1-based normalization as in Wikipedia
    D = [[zernike(index_style(j); coord=:polar)(r,p) for (r,p) in zip(rho, phi)] for j in J ]
    # use the coefficients as defined on Wikipedia rather than the Zernike definition of the toolbox
    mod_coeff = [coef ./ normalization(index_style(j)) for (j,coef) in zip(J, coefficients)] # get_Zernike_normcoeff(index, j) for (j,coef) in zip(J, coefficients)] 
    return reduce(+,map(*,D, mod_coeff))
end

function get_zernike_index_style(sym)
    if sym == :OSA
        return Noll
    elseif sym == :Noll
        return OSA
    else
        error("Unknown Zernike index style")
    end
end

"""
    get_zernike_pupil(sz, pp, sampling)

calculates the phases in the pupil for a given set of aberrations as defined by `J` and `coefficients`.
By default this follows the OSA nomenclature. See the help file of `ZernikePolynomials.jl` for more information.
The complex-valued pupil (up to the pupil border as defined by the `NA` in `pp`) is returned.

Arguments:
+ `sz`:  size of the real-space array
+ `pp`:  PSF parameter structure
+ `sampling`: pixelpitch in real space as NTuple

Example:
```jdoctest
julia> using PointSpreadFunctions, FFTW

julia> aberr = PointSpreadFunctions.Aberrations([Zernike_Spherical, Zernike_ObliqueAstigmatism],[0.1, 0.2])
Aberrations([12, 3], [0.1, 0.2], :OSA)

julia> pp = PSFParams(580.0, 1.4, 1.518; aberrations=aberr)
PSFParams(580.0, 1.4, 1.518, Float32, ModeWidefield, PointSpreadFunctions.pol_scalar, PointSpreadFunctions.var"#42#43"(), PointSpreadFunctions.MethodPropagateIterative, nothing, Aberrations([12, 3], [0.1, 0.2], :OSA), nothing)

julia> sz = (10,10,64)
(10, 10, 64)

julia> sampling=(190,190,100)
(190, 190, 100)

julia> PointSpreadFunctions.get_zernike_pupil(sz,pp,sampling)
10×10 Matrix{ComplexF64}:
 1.0+0.0im        1.0+0.0im               1.0+0.0im             1.0+0.0im               1.0+0.0im       …          1.0+0.0im             1.0+0.0im               1.0+0.0im             1.0+0.0im
 1.0+0.0im        1.0+0.0im               1.0+0.0im       -0.977799-0.209546im     0.297025+0.95487im         0.478105-0.878303im   0.238146-0.971229im          1.0+0.0im             1.0+0.0im
 1.0+0.0im        1.0+0.0im         -0.989823+0.142305im   0.389073+0.921207im     0.983723+0.179694im     -0.00466964-0.999989im  -0.696362-0.717691im    -0.643318-0.765599im        1.0+0.0im
 1.0+0.0im  -0.977799-0.209546im     0.389073+0.921207im   0.831518+0.555499im     0.894807+0.446453im        0.759688-0.650288im  -0.180764-0.983527im    -0.696362-0.717691im   0.238146-0.971229im
 1.0+0.0im   0.297025+0.95487im      0.983723+0.179694im   0.894807+0.446453im     0.563395+0.826187im        0.926073+0.377344im   0.759688-0.650288im  -0.00466964-0.999989im   0.478105-0.878303im
 1.0+0.0im   0.991619-0.129199im     0.790818-0.612051im   0.994312+0.106505im     0.520607+0.853797im  …     0.520607+0.853797im   0.994312+0.106505im     0.790818-0.612051im   0.991619-0.129199im
 1.0+0.0im   0.478105-0.878303im  -0.00466964-0.999989im   0.759688-0.650288im     0.926073+0.377344im        0.563395+0.826187im   0.894807+0.446453im     0.983723+0.179694im   0.297025+0.95487im
 1.0+0.0im   0.238146-0.971229im    -0.696362-0.717691im  -0.180764-0.983527im     0.759688-0.650288im        0.894807+0.446453im   0.831518+0.555499im     0.389073+0.921207im  -0.977799-0.209546im
 1.0+0.0im        1.0+0.0im         -0.643318-0.765599im  -0.696362-0.717691im  -0.00466964-0.999989im        0.983723+0.179694im   0.389073+0.921207im    -0.989823+0.142305im        1.0+0.0im
 1.0+0.0im        1.0+0.0im               1.0+0.0im        0.238146-0.971229im     0.478105-0.878303im        0.297025+0.95487im   -0.977799-0.209546im          1.0+0.0im             1.0+0.0im

```
"""
function get_zernike_pupil(sz, pp, sampling) 
    #cispi.(2 .*get_zernike_pupil_phase(sz, pp, sampling))
    cis.(get_zernike_pupil_phase(sz, pp, sampling))
end
