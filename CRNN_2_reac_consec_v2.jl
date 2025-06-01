using Flux
using Plots
using LinearAlgebra
using Statistics
using Random
using DifferentialEquations
using SciMLSensitivity

# Définition du Chemical Reaction Neural Network (CRNN)
mutable struct ChemicalReactionNN
    num_species::Int
    num_reactions::Int
    alpha::Matrix{Float32}    # Coefficients stoechiométriques des réactifs
    beta::Matrix{Float32}     # Coefficients stoechiométriques des produits
    k::Vector{Float32}        # Coefficients de vitesse des réactions
end

# Constructeur avec paramètres aléatoires
function ChemicalReactionNN(num_species::Int, num_reactions::Int)
    alpha = Float32.(rand(num_reactions, num_species))
    beta = Float32.(rand(num_reactions, num_species))
    k = Float32.(rand(num_reactions))
    return ChemicalReactionNN(num_species, num_reactions, alpha, beta, k)
end

# Fonction pour calculer les taux de réaction selon la loi d'action de masse
function reaction_rates(crnn::ChemicalReactionNN, c::Vector{Float32})
    rates = ones(Float32, crnn.num_reactions)
    for r in 1:crnn.num_reactions
        for s in 1:crnn.num_species
            if crnn.alpha[r, s] > 0
                rates[r] *= c[s]^crnn.alpha[r, s]
            end
        end
    end
    return rates .* crnn.k
end

# Fonction qui définit le système d'équations différentielles
function crnn_ode!(du, u, p, t)
    crnn = p
    c = Float32.(u)
    rates = reaction_rates(crnn, c)
    
    # Calculer les variations de concentration
    fill!(du, 0.0)
    for r in 1:crnn.num_reactions
        for s in 1:crnn.num_species
            du[s] += rates[r] * (crnn.beta[r, s] - crnn.alpha[r, s])
        end
    end
    
    # Éviter les concentrations négatives
    for i in eachindex(du)
        if u[i] + du[i] < 0
            du[i] = -u[i]
        end
    end
end

# Simulation avec DifferentialEquations.jl (compatible avec la différentiation)
function simulate_differentiable(crnn::ChemicalReactionNN, c_init::Vector{Float32}, tspan=(0.0f0, 1.0f0), saveat=nothing)
    if saveat === nothing
        saveat = range(tspan[1], tspan[2], length=101)
    end
    
    prob = ODEProblem(crnn_ode!, c_init, tspan, crnn)
    sol = solve(prob, Tsit5(), saveat=saveat, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    
    # Convertir la solution en matrice
    return Array(sol)'
end

# Version simple pour la visualisation
function simulate(crnn::ChemicalReactionNN, c_init::Vector{Float32}, steps::Int=100, dt::Float32=0.01f0)
    tspan = (0.0f0, Float32(steps * dt))
    saveat = range(0.0f0, tspan[2], length=steps+1)
    return simulate_differentiable(crnn, c_init, tspan, saveat)
end

# Fonction d'entraînement simplifiée
function train_crnn(crnn::ChemicalReactionNN, c_init::Vector{Float32}, target::Matrix{Float32},
                    epochs::Int=300, lr::Float32=0.01f0)

    # Optimiseur
    opt_state = Flux.setup(Adam(lr), crnn)
    
    # Temps correspondant au target
    tspan = (0.0f0, Float32(size(target, 1) - 1) * 0.1f0)
    saveat = range(tspan[1], tspan[2], length=size(target, 1))

    # Historique des pertes
    losses = Float32[]

    for epoch in 1:epochs
        # Calcul de la perte et des gradients
        loss, grads = Flux.withgradient(crnn) do model
            trajectory = simulate_differentiable(model, c_init, tspan, saveat)
            return Flux.mse(trajectory, target)
        end

        # Mettre à jour les paramètres
        Flux.update!(opt_state, crnn, grads[1])

        # Contraindre les paramètres pour éviter les instabilités
        crnn.alpha .= clamp.(crnn.alpha, 0, 2)
        crnn.beta .= clamp.(crnn.beta, 0, 2)
        crnn.k .= clamp.(crnn.k, 0, 5)

        # Enregistrer la perte
        if epoch % 30 == 0
            push!(losses, loss)
            println("Epoch $epoch, Loss: $loss")
        end
    end

    return losses
end

# Génération de données cibles oscillantes
function generate_target_oscillation(num_species::Int, steps::Int=100)
    t = range(0, 4π, length=steps+1)
    target = zeros(Float32, steps+1, num_species)

    for i in 1:num_species
        phase = (i-1) * π/num_species
        target[:, i] = 0.4f0 .* sin.(t .+ phase) .+ 0.5f0
    end

    return target
end

# Fonction principale
function main()
    Random.seed!(42)

    # Définir le réseau
    num_species = 3
    num_reactions = 4
    crnn = ChemicalReactionNN(num_species, num_reactions)

    # Concentration initiale
    c_init = Float32[0.5, 0.5, 0.5]

    # Générer des données cibles oscillantes
    target = generate_target_oscillation(num_species, 50)  # 51 points

    # Test de la simulation différentiable
    println("Test de la simulation différentiable...")
    tspan = (0.0f0, 5.0f0)
    saveat = range(0.0f0, 5.0f0, length=51)
    test_traj = simulate_differentiable(crnn, c_init, tspan, saveat)
    println("Simulation test réussie, taille: ", size(test_traj))

    # Entraîner le modèle
    println("Début de l'entraînement...")
    losses = train_crnn(crnn, c_init, target)

    # Visualiser les résultats
    println("Génération des graphiques...")
    final_trajectory = simulate(crnn, c_init, 50)

    # Trajectoires des espèces
    p1 = plot(final_trajectory, label=["Espèce $i" for i in 1:num_species]',
              xlabel="Pas de temps", ylabel="Concentration",
              title="Dynamique du système CRNN après entraînement")
    plot!(p1, target[:, 1], linestyle=:dash, color=:black, label="Cible")

    # Évolution de la perte
    p2 = plot(30:30:300, losses, xlabel="Epochs", ylabel="Perte",
              title="Évolution de la perte pendant l'entraînement")

    # Afficher les deux graphiques
    plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig("crnn_results.png")

    # Afficher les matrices de stœchiométrie apprises
    println("\nMatrice des réactifs (alpha):")
    display(crnn.alpha)

    println("\nMatrice des produits (beta):")
    display(crnn.beta)

    println("\nTaux de réaction (k):")
    display(crnn.k)
end

# Exécuter la fonction principale
main()