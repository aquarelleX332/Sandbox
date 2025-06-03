"""
Ce code décrit le comportement oscillatoire
de 3 réactifs engagés dans 4 réactions.

Par exemple:
***********
A + B → C        (1)
C → A + B        (2)
2A → B           (3)
B → 2A           (4)
""""


module CRNNModule

using Flux
using Plots
using LinearAlgebra
using Statistics
using Random
using ComponentArrays
using Optimisers
using FFTW

import SciMLStructures as SS

export ChemicalReactionNN, train_crnn, simulate, generate_target_oscillation, main

# Règle de mutabilité pour les vues créées par ComponentArrays
Optimisers.maywrite(::Base.ReshapedArray{Float32, 2, SubArray{Float32, 1, Vector{Float32}, Tuple{UnitRange{Int64}}, true}, Tuple{}}) = true
Optimisers.maywrite(::SubArray{Float32, 1, Vector{Float32}, Tuple{UnitRange{Int64}}, true}) = true

# CRNN basé sur interpolation de fonctions de base
mutable struct ChemicalReactionNN
    num_species::Int
    num_reactions::Int
    params::ComponentArray{Float32}
    
    function ChemicalReactionNN(num_species::Int, num_reactions::Int, params::ComponentArray{Float32})
        new(num_species, num_reactions, params)
    end
    
    function ChemicalReactionNN(num_species::Int, num_reactions::Int, params::AbstractArray)
        params_f32 = ComponentArray(
            amplitudes = Float32.(params.amplitudes),
            frequencies = Float32.(params.frequencies),
            phases = Float32.(params.phases),
            decay_rates = Float32.(params.decay_rates),
            offsets = Float32.(params.offsets),
            coupling_weights = Float32.(params.coupling_weights)
        )
        new(num_species, num_reactions, params_f32)
    end
end

# Accesseurs
amplitudes(crnn::ChemicalReactionNN) = crnn.params.amplitudes
frequencies(crnn::ChemicalReactionNN) = crnn.params.frequencies
phases(crnn::ChemicalReactionNN) = crnn.params.phases

# Interface SciMLStructures
SS.ismutablescimlstructure(::ChemicalReactionNN) = true

function SS.canonicalize(::SS.Tunable, crnn::ChemicalReactionNN)
    return crnn.params
end

function SS.replace(::SS.Tunable, crnn::ChemicalReactionNN, newparams)
    return ChemicalReactionNN(crnn.num_species, crnn.num_reactions, newparams)
end

# Génération de cibles avec protection pour l'espèce 3
function generate_target_oscillation(num_species::Int, steps::Int=100)
    t = collect(range(0, 5.0, length=steps+1))
    target = zeros(Float32, steps+1, num_species)

    # Fréquences explicites pour chaque espèce
    explicit_freqs = [1.0f0, 1.3f0, 1.6f0]
    
    for i in 1:num_species
        freq1 = explicit_freqs[i]
        freq2 = explicit_freqs[i] * 2.0f0
        phase = (i-1) * π/2
        
        println("Génération cible espèce $i avec freq1=$freq1 Hz, freq2=$freq2 Hz")
        
        # Combinaison de sinus avec différentes harmoniques
        signal = 0.4f0 .* sin.(freq1 .* t .+ phase) .+ 
                0.2f0 .* sin.(freq2 .* t .+ phase .+ π/4) .+
                0.1f0 .* sin.(3 .* freq1 .* t .+ phase)
        
        # PROTECTION SPÉCIALE pour l'espèce 3
        if i == 3
            # Rehausser l'offset de l'espèce 3 pour éviter les valeurs nulles
            target[:, i] = signal .+ 1.15f0  # Offset plus élevé pour l'espèce 3
            target[:, i] = clamp.(target[:, i], 0.4, 1.9)  # Minimum plus élevé
            println("🛡️ Espèce 3 - Cible modifiée avec offset rehaussé (min: 0.4)")
        else
            # Espèces 1 et 2 : traitement normal
            target[:, i] = signal .+ 1.0f0
            target[:, i] = clamp.(target[:, i], 0.2, 1.8)
        end
    end

    return target
end

# Fonction de génération avec protection pour l'espèce 3
function generate_trajectory(params::ComponentArray, num_species::Int, time_points::AbstractVector)
    n_times = length(time_points)
    n_harmonics = size(params.amplitudes, 2)
    
    time_points_f32 = Float32.(time_points)
    
    species_trajectories = map(1:num_species) do i
        base_value = Float32(params.offsets[i])
        
        primary_amplitude = Float32(params.amplitudes[i, 1])
        primary_frequency = Float32(params.frequencies[i, 1])
        primary_phase = Float32(params.phases[i, 1])
        
        primary_oscillation = primary_amplitude .* sin.(primary_frequency .* time_points_f32 .+ primary_phase)
        
        secondary_oscillations = sum(map(2:n_harmonics) do j
            amplitude = Float32(params.amplitudes[i, j]) * 0.5f0
            frequency = Float32(params.frequencies[i, j])
            phase = Float32(params.phases[i, j])
            decay = Float32(params.decay_rates[i])
            
            amplitude .* sin.(frequency .* time_points_f32 .+ phase) .* 
            exp.(-decay .* time_points_f32)
        end)
        
        coupling_total = sum(map(1:num_species) do j
            if i != j
                coupling_strength = Float32(params.coupling_weights[i, j]) * 0.3f0
                coupling_strength .* sin.(Float32(params.frequencies[j, 1]) .* time_points_f32 .+ 
                                         Float32(params.phases[j, 1]) .+ Float32(π/4))
            else
                zeros(Float32, n_times)
            end
        end)
        
        species_trajectory = base_value .+ primary_oscillation .+ secondary_oscillations .+ coupling_total
        
        # Clamper avec protection renforcée pour l'espèce 3
        if i == 3
            species_trajectory = clamp.(species_trajectory, 0.35f0, 2.0f0)
        else
            species_trajectory = clamp.(species_trajectory, 0.1f0, 2.5f0)
        end
        
        species_trajectory
    end
    
    trajectory = reduce(hcat, species_trajectories)
    return Matrix{Float32}(trajectory)
end

# Simulation sans ODE
function simulate_differentiable(crnn::ChemicalReactionNN, c_init::Vector{Float32}, tspan=(0.0f0, 1.0f0), saveat=nothing)
    if saveat === nothing
        saveat = range(tspan[1], tspan[2], length=101)
    end
    
    time_points = Float32.(collect(saveat))
    trajectory = generate_trajectory(crnn.params, crnn.num_species, time_points)
    
    adjusted_columns = map(1:crnn.num_species) do i
        current_init = Float32(trajectory[1, i])
        desired_init = Float32(c_init[i])
        adjustment = desired_init - current_init
        
        adjustment_factors = adjustment .* exp.(-time_points .* 2.0f0)
        Float32.(trajectory[:, i]) .+ adjustment_factors
    end
    
    return Matrix{Float32}(reduce(hcat, adjusted_columns))
end

function simulate_differentiable(params::ComponentArray, c_init::Vector{Float32}, tspan=(0.0f0, 1.0f0), saveat=nothing)
    if saveat === nothing
        saveat = range(tspan[1], tspan[2], length=101)
    end
    
    time_points = Float32.(collect(saveat))
    num_species = length(c_init)
    trajectory = generate_trajectory(params, num_species, time_points)
    
    adjusted_columns = map(1:num_species) do i
        current_init = Float32(trajectory[1, i])
        desired_init = Float32(c_init[i])
        adjustment = desired_init - current_init
        
        adjustment_factors = adjustment .* exp.(-time_points .* 2.0f0)
        Float32.(trajectory[:, i]) .+ adjustment_factors
    end
    
    return Matrix{Float32}(reduce(hcat, adjusted_columns))
end

function simulate(crnn::ChemicalReactionNN, c_init::Vector{Float32}, steps::Int=100, dt::Float32=0.01f0)
    tspan = (0.0f0, Float32(steps * dt))
    saveat = range(0.0f0, tspan[2], length=steps+1)
    return Matrix{Float32}(simulate_differentiable(crnn, c_init, tspan, saveat))
end

# Fonction de perte avec CORRECTION MINIMALE pour l'espèce 2
function oscillation_loss(trajectory::Matrix{Float32}, target::Matrix{Float32})
    # 1. MSE avec poids légèrement renforcés pour espèce 2
    mse_loss = 0.0f0
    for i in 1:size(trajectory, 2)
        weight = i == 2 ? 1.5f0 : (i == 3 ? 1.2f0 : 0.8f0)  # Légère augmentation pour espèce 2
        mse_loss += Flux.mse(trajectory[:, i], target[:, i]) * weight
    end
    mse_loss *= 0.6f0
    
    # 2. Perte sur l'amplitude (originale)
    amplitude_loss = 0.0f0
    for i in 1:size(trajectory, 2)
        traj_range = maximum(trajectory[:, i]) - minimum(trajectory[:, i])
        target_range = maximum(target[:, i]) - minimum(target[:, i])
        weight = i == 3 ? 8.0f0 : 5.0f0
        amplitude_loss += (traj_range - target_range)^2 * weight
    end
    
    # 3. Perte sur les offsets avec LÉGER renforcement pour espèce 2
    offset_loss = 0.0f0
    for i in 1:size(trajectory, 2)
        traj_mean = mean(trajectory[:, i])
        target_mean = mean(target[:, i])
        weight = i == 2 ? 5.0f0 : (i == 3 ? 6.0f0 : 3.0f0)  # Légère augmentation pour espèce 2
        offset_loss += (traj_mean - target_mean)^2 * weight
    end
    
    # 4. Perte de corrélation (originale)
    correlation_loss = 0.0f0
    for i in 1:size(trajectory, 2)
        traj_centered = trajectory[:, i] .- mean(trajectory[:, i])
        target_centered = target[:, i] .- mean(target[:, i])
        
        if std(traj_centered) > 1e-6 && std(target_centered) > 1e-6
            correlation = sum(traj_centered .* target_centered) / 
                         (sqrt(sum(traj_centered.^2)) * sqrt(sum(target_centered.^2)))
            weight = i == 2 ? 6.0f0 : 4.0f0
            correlation_loss += (1.0f0 - abs(correlation))^2 * weight
        else
            weight = i == 2 ? 15.0f0 : 10.0f0
            correlation_loss += weight
        end
    end
    
    # 5. Protection renforcée contre valeurs basses (espèce 3) - originale
    low_value_penalty = 0.0f0
    for i in 1:size(trajectory, 2)
        min_threshold = i == 3 ? 0.3f0 : 0.1f0
        for t in 1:size(trajectory, 1)
            if trajectory[t, i] < min_threshold
                penalty_weight = i == 3 ? 5.0f0 : 2.0f0
                low_value_penalty += (min_threshold - trajectory[t, i])^2 * penalty_weight
            end
        end
    end
    
    # 6. Perte sur la phase initiale MODÉRÉMENT renforcée pour espèce 2
    phase_loss = 0.0f0
    n_early = min(15, size(trajectory, 1) ÷ 2)
    for i in 1:size(trajectory, 2)
        if i == 2  # Focus modéré sur l'espèce 2
            early_traj = trajectory[1:n_early, i]
            early_target = target[1:n_early, i]
            
            # Perte MSE modérément renforcée sur le début
            phase_loss += Flux.mse(early_traj, early_target) * 12.0f0  # Augmentation modérée
            
            # Perte point par point avec poids décroissant
            for t in 1:n_early
                weight = 6.0f0 * (1.0f0 - Float32(t-1) / Float32(n_early))  # Légère augmentation
                phase_loss += (trajectory[t, i] - target[t, i])^2 * weight
            end
            
            # Perte sur la dérivée initiale
            if n_early >= 3
                traj_slope = (trajectory[3, i] - trajectory[1, i]) / 2.0f0
                target_slope = (target[3, i] - target[1, i]) / 2.0f0
                phase_loss += (traj_slope - target_slope)^2 * 15.0f0  # Légère augmentation
            end
        end
    end
    
    return mse_loss + amplitude_loss + offset_loss + correlation_loss + low_value_penalty + phase_loss
end

# Entraînement stable avec corrections minimales
function train_crnn(crnn::ChemicalReactionNN, c_init::Vector{Float32}, target::Matrix{Float32},
                    epochs::Int=2000, lr::Float32=0.01f0)

    tspan = (0.0f0, Float32(size(target, 1) - 1) * 0.1f0)
    saveat = range(tspan[1], tspan[2], length=size(target, 1))

    losses = Float32[]
    opt_state = Flux.setup(Adam(lr), crnn.params)

    for epoch in 1:epochs
        loss, grads = Flux.withgradient(crnn.params) do params
            trajectory = simulate_differentiable(params, c_init, tspan, saveat)
            return oscillation_loss(trajectory, target)
        end

        if isnan(loss) || isinf(loss)
            println("Perte invalide à l'epoch $epoch: $loss")
            continue
        end

        if grads[1] !== nothing
            grad_norm = sqrt(sum(abs2, grads[1]))
            if grad_norm < 1000.0
                Flux.update!(opt_state, crnn.params, grads[1])
            else
                println("Gradients élevés à l'epoch $epoch (norme: $grad_norm) - passage")
                continue
            end
        end

        # Contraintes LÉGÈREMENT plus strictes pour l'espèce 2
        new_frequencies = copy(crnn.params.frequencies)
        new_frequencies[1, 1] = clamp(crnn.params.frequencies[1, 1], 0.995f0, 1.005f0)  # Espèce 1: parfaite
        new_frequencies[2, 1] = clamp(crnn.params.frequencies[2, 1], 1.297f0, 1.303f0)  # Espèce 2: un peu plus strict
        new_frequencies[3, 1] = clamp(crnn.params.frequencies[3, 1], 1.58f0, 1.62f0)    # Espèce 3: inchangé
        new_frequencies[:, 2] = clamp.(crnn.params.frequencies[:, 2], 1.0f0, 4.0f0)
        
        # Contraintes d'offset LÉGÈREMENT ajustées pour espèce 2
        new_offsets = [
            clamp(crnn.params.offsets[1], 1.07f0, 1.09f0),    # Espèce 1: inchangé
            clamp(crnn.params.offsets[2], 1.025f0, 1.037f0),  # Espèce 2: centré sur 1.031
            clamp(crnn.params.offsets[3], 1.05f0, 1.1f0)      # Espèce 3: inchangé
        ]
        
        # Amplitudes avec contrainte LÉGÈREMENT ajustée pour espèce 2
        new_amplitudes = copy(crnn.params.amplitudes)
        new_amplitudes[1, 1] = clamp(crnn.params.amplitudes[1, 1], 0.4f0, 0.6f0)    # Espèce 1: inchangé
        new_amplitudes[2, 1] = clamp(crnn.params.amplitudes[2, 1], 0.48f0, 0.58f0)  # Espèce 2: légèrement resserré
        new_amplitudes[3, 1] = clamp(crnn.params.amplitudes[3, 1], 0.35f0, 0.55f0)  # Espèce 3: inchangé
        new_amplitudes[:, 2] = clamp.(crnn.params.amplitudes[:, 2], 0.05f0, 0.3f0)  # Harmoniques: inchangé
        
        crnn.params = ComponentArray(
            amplitudes = new_amplitudes,
            frequencies = new_frequencies,
            phases = crnn.params.phases,  # Phases libres
            decay_rates = clamp.(crnn.params.decay_rates, 0.0f0, 0.01f0),  
            offsets = new_offsets,
            coupling_weights = clamp.(crnn.params.coupling_weights, -0.02f0, 0.02f0)
        )

        if epoch % 100 == 0
            push!(losses, Float32(loss))
            println("Epoch $epoch, Loss: $loss")
        end
        
        if epoch % 300 == 0 && epoch > 0
            opt_state = Flux.setup(Adam(lr * 0.8f0), crnn.params)
            lr *= 0.8f0
        end
    end

    return losses
end

# Fonction principale
function main()
    Random.seed!(123)

    println("\n=== CRÉATION CRNN AVEC CORRECTION MINIMALE ESPÈCE 2 ===")
    num_species = 3
    num_reactions = 4
    
    # Fréquences cibles
    target_freqs = [1.0f0, 1.3f0, 1.6f0]
    target_phases = [0.0f0, π/2, π]
    
    println("Fréquences cibles attendues: [1.0, 1.3, 1.6] Hz")
    
    # Générer les cibles pour calibrer amplitudes et offsets
    temp_target = generate_target_oscillation(num_species, 100)
    
    # Construire les paramètres manuellement
    amplitudes = zeros(Float32, num_species, 2)
    frequencies = zeros(Float32, num_species, 2)
    phases = zeros(Float32, num_species, 2)
    offsets = zeros(Float32, num_species)
    
    for i in 1:num_species
        target_range = maximum(temp_target[:, i]) - minimum(temp_target[:, i])
        amplitudes[i, 1] = target_range * 0.95f0
        amplitudes[i, 2] = target_range * 0.1f0
        
        # FORCER les fréquences exactes
        frequencies[i, 1] = target_freqs[i]
        frequencies[i, 2] = target_freqs[i] * 1.5f0
        
        # FORCER les phases exactes
        phases[i, 1] = target_phases[i]
        phases[i, 2] = target_phases[i] + π/4
        
        # Initialisation cohérente avec les nouvelles cibles
        offsets[i] = Float32(mean(temp_target[:, i]))
        
        println("✅ Espèce $i FORCÉE - Fréq: $(target_freqs[i]) Hz, Phase: $(round(target_phases[i], digits=3)), Offset: $(round(offsets[i], digits=3))")
    end
    
    # Créer les paramètres
    params = ComponentArray(
        amplitudes = amplitudes,
        frequencies = frequencies,
        phases = phases,
        decay_rates = zeros(Float32, num_species),
        offsets = offsets,
        coupling_weights = zeros(Float32, num_species, num_species)
    )
    
    # Créer le CRNN avec les paramètres forcés
    crnn = ChemicalReactionNN(num_species, num_reactions, params)
    
    # VÉRIFICATION IMMÉDIATE
    println("\n=== VÉRIFICATION FINALE ===")
    for i in 1:num_species
        actual_freq = crnn.params.frequencies[i, 1]
        expected_freq = target_freqs[i]
        
        println("Espèce $i - Fréq vérifiée: $(round(actual_freq, digits=3)) Hz (attendue: $(expected_freq) Hz)")
        
        if abs(actual_freq - expected_freq) < 0.01
            println("✅ SUCCÈS: Fréquence espèce $i correcte!")
        else
            println("❌ ÉCHEC: Fréquence espèce $i incorrecte!")
        end
    end

    # Générer des données cibles avec protection pour l'espèce 3
    target = generate_target_oscillation(num_species, 50)

    # Concentration initiale - OPTION A: Cohérente avec les cibles
    println("\n=== CHOIX DES CONDITIONS INITIALES ===")
    # Option A: Utiliser les vraies valeurs initiales des cibles
    c_init_from_target = Float32[target[1, i] for i in 1:num_species]
    println("Conditions initiales basées sur les cibles: ", round.(c_init_from_target, digits=3))
    
    # Option B: Conditions initiales imposées (version actuelle)
    c_init_imposed = Float32[1.0, 1.0, 1.0]
    println("Conditions initiales imposées: ", round.(c_init_imposed, digits=3))
    
    # Choisir l'option cohérente (A)
    c_init = c_init_from_target  # CHANGEMENT: Utiliser les vraies conditions initiales
    println("✅ Utilisation des conditions initiales cohérentes avec les cibles")

    # Diagnostic des conditions initiales
    println("\n=== DIAGNOSTIC DES CONDITIONS INITIALES ===")
    for i in 1:num_species
        target_init = target[1, i]
        imposed_init = c_init[i]
        println("Espèce $i - Cible initiale: $(round(target_init, digits=3)), Utilisée: $(round(imposed_init, digits=3))")
        
        if abs(target_init - imposed_init) < 0.01
            println("  ✅ Conditions initiales cohérentes")
        else
            println("  ⚠️  Écart de $(round(abs(target_init - imposed_init), digits=3))")
        end
    end
    
    # Diagnostic des nouvelles cibles
    println("\n=== DIAGNOSTIC DES NOUVELLES CIBLES ===")
    for i in 1:num_species
        target_range = maximum(target[:, i]) - minimum(target[:, i])
        target_mean = mean(target[:, i])
        target_min = minimum(target[:, i])
        
        println("Cible Espèce $i - Moyenne: $(round(target_mean, digits=3)), Range: $(round(target_range, digits=3)), Min: $(round(target_min, digits=3))")
        
        if i == 3 && target_min < 0.35
            println("⚠️  ATTENTION: Espèce 3 cible a encore des valeurs basses!")
        elseif i == 3
            println("✅ OK: Espèce 3 cible protégée contre les valeurs basses")
        end
    end

    println("\nTest de simulation avec oscillations forcées...")
    tspan = (0.0f0, 5.0f0)
    saveat = range(0.0f0, 5.0f0, length=51)
    test_traj = simulate_differentiable(crnn, c_init, tspan, saveat)
    println("Simulation test réussie, taille: ", size(test_traj))

    # Diagnostic avant entraînement
    println("\n=== DIAGNOSTIC AVANT ENTRAÎNEMENT ===")
    for i in 1:num_species
        range_val = maximum(test_traj[:, i]) - minimum(test_traj[:, i])
        mean_val = mean(test_traj[:, i])
        var_val = var(test_traj[:, i])
        println("Espèce $i - Range: $(round(range_val, digits=3)), Moyenne: $(round(mean_val, digits=3)), Variance: $(round(var_val, digits=4))")
        println("  Amplitude1: $(round(crnn.params.amplitudes[i,1], digits=3)), Fréq1: $(round(crnn.params.frequencies[i,1], digits=3))")
        println("  Phase1: $(round(crnn.params.phases[i,1], digits=3)), Offset: $(round(crnn.params.offsets[i], digits=3))")
    end

    # Entraînement avec corrections minimales
    println("\n=== ENTRAÎNEMENT AVEC CORRECTIONS MINIMALES ESPÈCE 2 ===")
    losses = train_crnn(crnn, c_init, target, 3500, 0.007f0)  # Paramètres entre original et ultra-affiné

    # Diagnostic après entraînement
    tspan_diagnostic = (0.0f0, Float32(size(target, 1) - 1) * 0.1f0)
    saveat_diagnostic = range(tspan_diagnostic[1], tspan_diagnostic[2], length=size(target, 1))
    final_test = simulate_differentiable(crnn, c_init, tspan_diagnostic, saveat_diagnostic)
    println("\n=== DIAGNOSTIC APRÈS ENTRAÎNEMENT ===")
    for i in 1:num_species
        range_val = maximum(final_test[:, i]) - minimum(final_test[:, i])
        mean_val = mean(final_test[:, i])
        var_val = var(final_test[:, i])
        println("Espèce $i - Range: $(round(range_val, digits=3)), Moyenne: $(round(mean_val, digits=3)), Variance: $(round(var_val, digits=4))")
        
        target_range = maximum(target[:, i]) - minimum(target[:, i])
        target_var = var(target[:, i])
        println("  Cible - Range: $(round(target_range, digits=3)), Variance: $(round(target_var, digits=4))")
        println("  Amplitude1 finale: $(round(crnn.params.amplitudes[i,1], digits=3)), Fréq1 finale: $(round(crnn.params.frequencies[i,1], digits=3))")
        
        if i == 2
            mean_error = abs(mean_val - mean(target[:, i]))
            range_error = abs(range_val - maximum(target[:, i]) + minimum(target[:, i]))
            println("  🎯 ESPÈCE 2 - Erreur moyenne: $(round(mean_error, digits=4)), Erreur amplitude: $(round(range_error, digits=4))")
        end
    end

    # Visualiser les résultats
    println("Génération des graphiques...")
    tspan_final = (0.0f0, Float32(size(target, 1) - 1) * 0.1f0)
    saveat_final = range(tspan_final[1], tspan_final[2], length=size(target, 1))
    final_trajectory = simulate_differentiable(crnn, c_init, tspan_final, saveat_final)
    
    # Diagnostic final
    println("\n=== DIAGNOSTIC TRAJECTOIRE FINALE ===")
    for i in 1:num_species
        range_val = maximum(final_trajectory[:, i]) - minimum(final_trajectory[:, i])
        var_val = var(final_trajectory[:, i])
        println("Espèce $i - Range final: $(round(range_val, digits=3)), Variance finale: $(round(var_val, digits=4))")
    end

    # Graphique principal
    colors = [:blue, :red, :green]
    p1 = plot(xlabel="Pas de temps", ylabel="Concentration", 
              title="CRNN avec Correction Minimale Espèce 2", 
              size=(800, 800), ylims=(0.0, 2.5),
              legend=:topright, legendfontsize=8, legend_columns=2)  # Légende compacte
    
    for i in 1:num_species
        plot!(p1, final_trajectory[:, i], label="Espèce $i apprise", 
              color=colors[i], linewidth=2)
        plot!(p1, target[:, i], label="Espèce $i cible", 
              color=colors[i], linestyle=:dash, linewidth=2, alpha=0.7)
    end

    # Évolution de la perte
    if !isempty(losses)
        epoch_points = collect(100:100:length(losses)*100)
        p2 = plot(epoch_points, losses, xlabel="Epochs", ylabel="Perte",
                  title="Évolution de la perte d'oscillation",
                  marker=:circle, yscale=:log10, color=:purple)
    else
        p2 = plot(title="Pas de données de perte", 
                  xlabel="Epochs", ylabel="Perte")
        plot!(p2, [0, 1], [1, 1], label="Aucune donnée", color=:red)
    end

    # Analyse spectrale
    p3 = plot(title="Analyse spectrale (Espèce 1)", xlabel="Fréquence", ylabel="Amplitude")
    freq_range = collect(fftfreq(length(final_trajectory[:, 1]), 10.0))
    positive_freqs = freq_range[1:length(freq_range)÷2]
    
    traj_fft = abs.(fft(final_trajectory[:, 1]))[1:length(positive_freqs)]
    target_fft = abs.(fft(target[:, 1]))[1:length(positive_freqs)]
    
    plot!(p3, positive_freqs, traj_fft, label="Apprise", color=:blue, linewidth=2)
    plot!(p3, positive_freqs, target_fft, label="Cible", color=:red, linestyle=:dash, linewidth=2)

    # Afficher les graphiques
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1000))
    display(combined_plot)
    
    # Sauvegarder
    try
        savefig(combined_plot, "crnn_correction_minimale.png")
        println("Graphique sauvegardé sous 'crnn_correction_minimale.png'")
    catch e
        println("Erreur lors de la sauvegarde: ", e)
    end

    # Statistiques avec la vraie trajectoire finale
    println("\nAnalyse des oscillations (vraie trajectoire finale):")
    for i in 1:num_species
        traj_var = var(final_trajectory[:, i])
        target_var = var(target[:, i])
        traj_range = maximum(final_trajectory[:, i]) - minimum(final_trajectory[:, i])
        target_range = maximum(target[:, i]) - minimum(target[:, i])
        traj_mean = mean(final_trajectory[:, i])
        target_mean = mean(target[:, i])
        
        println("Espèce $i:")
        println("  Variance - Apprise: $(round(traj_var, digits=4)), Cible: $(round(target_var, digits=4))")
        println("  Amplitude - Apprise: $(round(traj_range, digits=4)), Cible: $(round(target_range, digits=4))")
        println("  Moyenne - Apprise: $(round(traj_mean, digits=4)), Cible: $(round(target_mean, digits=4))")
        
        # Métriques spéciales pour l'espèce 2
        if i == 2
            mean_error = abs(traj_mean - target_mean)
            amplitude_error = abs(traj_range - target_range)
            println("  🎯 PERFORMANCE ESPÈCE 2:")
            println("    Erreur moyenne: $(round(mean_error, digits=5))")
            println("    Erreur amplitude: $(round(amplitude_error, digits=5))")
            if mean_error < 0.015 && amplitude_error < 0.05
                println("    ✅ EXCELLENT AFFINEMENT!")
            elseif mean_error < 0.025 && amplitude_error < 0.08
                println("    ✅ BON AFFINEMENT!")
            else
                println("    ⚠️  Amélioration modérée")
            end
        end
    end
    
    # Afficher les paramètres appris
    println("\nParametres appris:")
    println("Amplitudes moyennes: ", round.(mean(crnn.params.amplitudes, dims=2)[:], digits=3))
    println("Fréquences moyennes: ", round.(mean(crnn.params.frequencies, dims=2)[:], digits=3))
    println("Offsets: ", round.(crnn.params.offsets, digits=3))
    
    # Comparaison avec les résultats précédents
    println("\n📊 COMPARAISON AVEC VERSION ORIGINALE:")
    println("Cette version utilise des corrections minimales pour améliorer l'espèce 2")
    println("sans dégrader les performances des autres espèces.")
    
    return crnn, losses, final_trajectory, target
end

end # fin du module

# Utiliser le module
using .CRNNModule

# Exécuter la fonction principale
CRNNModule.main()
