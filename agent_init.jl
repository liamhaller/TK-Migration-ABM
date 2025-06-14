# Necessary Julia packages
# You might need to install them first using Pkg.add("PackageName")
using Agents
using Random
using Distributions
using DataFrames
using Graphs # For explicit graph manipulation if needed, though Agents.jl handles most.

# --- Agent Definition (Updated Syntax) ---
@agent struct Person(GraphAgent) # Inherit from GraphAgent
    # Demographic Attributes (age, gender, etc. - all your existing custom fields)
    # IMPORTANT: Do NOT re-declare `id` or `pos`. They come from GraphAgent.
    # `vel` will also no longer be present, which is correct for GraphSpace.
    age::Float64
    gender::Symbol
    province::Int
    education::Int
    employment_status::Symbol
    income::Float64
    has_family_abroad::Bool
    migration_stage::Symbol
    time_in_current_stage::Float64
    behavioral_attitude::Float64
    subjective_norm::Float64
    perceived_behavioral_control::Float64
    actual_behavioral_control::Float64
    V_score::Float64
    perceived_political_instability_turkey::Float64
    perceived_economic_conditions_turkey::Float64
    perceived_economic_conditions_germany::Float64
    perceived_policy_complexity::Float64
    skill_level::Symbol
    age_of_interest::Float64
end

# --- Helper Functions for Initialization & TPB ---

# Placeholder: Determine skill level based on agent attributes
# You'll need to refine this based on your specific definitions for high/low skill
function get_skill_level(education::Int, employment_status::Symbol)::Symbol
    # Example logic
    if education >= 16 && (employment_status == :white_collar || employment_status == :self_employment) # Assuming 16+ years is university degree
        return :high_skill
    else
        return :low_medium_skill
    end
end

# Placeholder: Generate initial age of interest for migration consideration
# Based on ODD section 5.1 (Submodels -> Never Consider to Assessment Transition)
function generate_age_of_interest(skill_level::Symbol, min_age_consideration::Float64 = 15.0)::Float64
    if skill_level == :high_skill
        # Highly skilled: Mean = 20 years, Std Dev = 2 years, Min = 15 years [cite: 523]
        dist = Truncated(Normal(20, 2), min_age_consideration, Inf)
    else
        # Low/medium skill: Mean = 19 years, Std Dev = 4 years, Min = 15 years [cite: 522]
        dist = Truncated(Normal(19, 4), min_age_consideration, Inf)
    end
    return rand(dist)
end

# Placeholder: Calculate Behavioral Attitude (BA)
# BA = sum(belief_strength_i * outcome_evaluation_i)
# This needs to be fleshed out with specific outcomes and their evaluations
function calculate_BA(agent::Person, model::ABM)::Float64
    # Outcomes (pi * ei) could be:
    # 1. Economic improvement (Germany econ vs Turkey econ)
    # 2. Political stability (Germany vs Turkey)
    # 3. Personal development, etc.
    # These pi and ei are subjective.

    # Simplified example:
    econ_diff_eval = (agent.perceived_economic_conditions_germany - agent.perceived_economic_conditions_turkey) * model.ba_weight_econ
    pol_diff_eval = (model.political_stability_germany - agent.perceived_political_instability_turkey) * model.ba_weight_pol
    # A base aspiration component (wanderlust/structural desire) [cite: 270]
    base_aspiration = model.ba_base_aspiration

    ba_score = base_aspiration + econ_diff_eval + pol_diff_eval
    return clamp(ba_score, model.ba_min_val, model.ba_max_val) # Clamp to a reasonable range
end

# Placeholder: Calculate Subjective Norm (SN)
# SN = sum(normative_belief_i * motivation_to_comply_i)
# Influenced by family abroad and network
function calculate_SN(agent::Person, model::ABM)::Float64
    sn_score = model.sn_base_value # Base SN from general culture
    num_migrated_friends = 0
    if Graphs.nv(getfield(model, :space).graph) > 1 && !isempty(nearby_ids(agent,model)) # Check number of vertices in the graph
        for neighbor_id in nearby_ids(agent, model) # nearby_ids from Agents.jl gives connected agents
            neighbor = model[neighbor_id]
            if neighbor.migration_stage == :migrated
                num_migrated_friends += 1
            end
        end
    end

    if agent.has_family_abroad
        sn_score += model.sn_weight_family_abroad
    end
    sn_score += num_migrated_friends * model.sn_weight_migrated_friend

    # Perceived social pressure from general environment (e.g. media - not explicitly in ODD agent attributes but implied in societal factors)
    sn_score += model.sn_societal_pressure_factor

    return clamp(sn_score, model.sn_min_val, model.sn_max_val)
end

# Placeholder: Calculate Perceived Behavioral Control (PBC)
# PBC = sum(control_belief_strength_i * perceived_power_of_control_factor_i)
# Influenced by agent's resources (education, income), policy perceptions, network support
function calculate_PBC(agent::Person, model::ABM)::Float64
    pbc_score = model.pbc_base_value

    # Self-efficacy based on personal attributes
    pbc_score += agent.education * model.pbc_weight_education
    pbc_score += agent.income * model.pbc_weight_income
    # Employment status might positively or negatively influence PBC
    if agent.employment_status == :unemployed
        pbc_score += model.pbc_effect_unemployed # could be negative if it means fewer resources
    elseif agent.employment_status == :white_collar
        pbc_score += model.pbc_effect_white_collar
    end

    # Controllability based on perceived external factors
    # Higher policy complexity should decrease PBC
    pbc_score -= agent.perceived_policy_complexity * model.pbc_weight_policy_complexity

    # Network effects (social capital, information)
    known_migrants_in_network = 0
    if nagents(model.space.graph) > 1 && !isempty(nearby_ids(agent,model))
        for neighbor_id in nearby_ids(agent, model)
            if model[neighbor_id].migration_stage == :migrated || model[neighbor_id].has_family_abroad
                known_migrants_in_network +=1 # simplified social capital proxy
            end
        end
    end
    pbc_score += known_migrants_in_network * model.pbc_weight_network_capital

    return clamp(pbc_score, model.pbc_min_val, model.pbc_max_val)
end

# Calculate Composite Intention Score V [cite: 187, 532]
# V(SN,PBC) = gamma * SN^alpha * PBC^beta
function calculate_V_score(sn::Float64, pbc::Float64, model::ABM)::Float64
    # Ensure SN and PBC are non-negative before power, or handle appropriately
    # ODD implies these scores are generally positive or represent strength.
    # Let's assume they are scaled to be positive.
    safe_sn = max(0.01, sn) # Avoid issues with 0^alpha if alpha is tricky
    safe_pbc = max(0.01, pbc)
    return model.gamma_V * (safe_sn^model.alpha_V) * (safe_pbc^model.beta_V)
end

# Placeholder: Calculate Actual Behavioral Control (ABC)
# Objective check against current German immigration policies
function calculate_ABC(agent::Person, model::ABM)::Float64
    # This is a score from 0 to 1 reflecting objective eligibility
    # For MVP, can be simplified. Full version requires policy data.

    # 1. Education/Skills Match (e.g., points system or binary)
    education_match = 0.0
    # EXAMPLE: Access current German policy for high-skilled workers from model.german_policy_settings
    required_edu_high_skill = model.german_policy_settings[:high_skill_education_threshold] # Placeholder
    if agent.skill_level == :high_skill && agent.education >= required_edu_high_skill
        education_match = 0.4 # Max 0.4 points for this
    elseif agent.skill_level == :low_medium_skill # Assume some low-skill pathways exist
        # Check against low_skill_education_threshold etc.
        education_match = 0.2 # Lower points for low_skill pathways
    end

    # 2. Job Offer (probabilistic) [cite: 210]
    # Probability could depend on agent.skill_level, agent.education, model.german_job_market_openness
    prob_job_offer = model.german_policy_settings[:base_job_offer_prob] # placeholder
    if agent.skill_level == :high_skill
        prob_job_offer *= model.german_policy_settings[:high_skill_job_multiplier]
    end
    job_offer_points = rand() < prob_job_offer ? 0.3 : 0.0 # Max 0.3 points

    # 3. Language Skills (simplified) [cite: 211]
    # Could be an agent attribute: agent.german_language_skill::Symbol (:none, :basic, :fluent)
    # Or inferred from education
    language_points = 0.0
    if agent.education > 12 # Simplified: more education -> higher chance of language skills
        language_points = rand() < model.german_policy_settings[:language_skill_prob_edu_gt_12] ? 0.15 : 0.0
    else
        language_points = rand() < model.german_policy_settings[:language_skill_prob_edu_lte_12] ? 0.05 : 0.0
    end

    # 4. Financial Requirements (simplified) [cite: 212]
    financial_points = 0.0
    if agent.income >= model.german_policy_settings[:financial_requirement_threshold]
        financial_points = 0.15
    end

    # Sum up points, max 1.0
    abc_score = clamp(education_match + job_offer_points + language_points + financial_points, 0.0, 1.0)
    return abc_score
end

# --- Social Network Formation ---
# Calculate demographic similarity
function demographic_similarity(agent1::Person, agent2::Person, model::ABM)::Float64
    s_age = 1.0 - (abs(agent1.age - agent2.age) / model.range_age)
    s_gender = agent1.gender == agent2.gender ? 1.0 : 0.0
    # Ethnicity assumed same for Turkish citizens in this model context as per ODD [cite: 507]
    # s_ethnicity = 1.0 # (w_ethnicity = 2/6)
    s_education = 1.0 - (abs(agent1.education - agent2.education) / model.range_education)

    # Weights[cite: 510]: age=3/6, ethnicity=2/6 (skipped), education=1/6.
    # Adjusting weights if ethnicity is skipped: age (3/4), education (1/4)
    # Or use original weights and assume ethnicity similarity = 1
    # ODD uses w_age = 3/6, w_ethnicity = 2/6, w_education = 1/6. Let's assume s_ethnicity = 1 implicitly.
    total_similarity = (model.w_age * s_age) +
                       (model.w_gender_for_homophily * s_gender) + # ODD mentions ethnicity, age, religion, education for homophily from [43] in source 574. Gender is not in that list, but is an agent attribute. Using a small weight.
                       (model.w_education * s_education) +
                       (model.w_ethnicity * 1.0) # Assuming s_ethnicity is 1.0
    
    return total_similarity / (model.w_age + model.w_gender_for_homophily + model.w_education + model.w_ethnicity) # Normalize
end

# Placeholder for Social Connectedness Index (SCI) factor
function provincial_connectivity_factor(province1::Int, province2::Int, model::ABM)::Float64
    # This should query your SCI data. For now, a placeholder.
    # Higher value means more connected. Needs normalization.
    # Example: return model.sci_matrix[province1, province2] / model.sci_max_value
    if province1 == province2
        return 1.0 # High connectivity within the same province
    else
        # Fake some decay with distance, then normalize
        # This is a very rough placeholder. YOU MUST REPLACE THIS.
        raw_sci = 1.0 / (1.0 + abs(province1 - province2))
        return raw_sci / 1.0 # Assuming max raw SCI is 1 for this placeholder
    end
end

# Function to form the initial network
function form_initial_network!(model::ABM)
    println("Forming initial social network...")
    # ... (your comments) ...

    # !!! ADD THESE TWO LINES BACK IN !!!
    agents_vector = collect(allagents(model)) # Convert the iterator to a Vector
    num_actual_agents = length(agents_vector) # Define num_actual_agents

    # Now the loops can use num_actual_agents and agents_vector
    for i in 1:num_actual_agents
        for j in (i+1):num_actual_agents 
            agent1 = agents_vector[i]    
            agent2 = agents_vector[j]    
            
            sim = demographic_similarity(agent1, agent2, model)
            prov_conn = provincial_connectivity_factor(agent1.province, agent2.province, model)
    
            connection_prob = sim * prov_conn * model.network_density_factor
            
            if rand() < connection_prob
                Graphs.add_edge!(model.space.graph, agent1.pos, agent2.pos)
            end
        end
    end
    println("Network formation complete. Number of edges: ", Graphs.ne(model.space.graph))
end


# --- Model Initialization ---
function initialize_model(;
    num_agents::Int = 1000, # Default number of agents
    # Parameters for agent initialization (distributions or fixed values)
    # These would ideally come from data files (TurkStat, EUMAGINE, TRANSMIT)
    # For now, using placeholders
    mean_age_init::Float64 = 35.0,
    std_dev_age_init::Float64 = 10.0,
    min_age_init::Float64 = 18.0,
    max_age_init::Float64 = 65.0, # Focus on labor migration age
    prob_male_init::Float64 = 0.5,
    num_provinces::Int = 81,
    mean_education_init::Float64 = 10.0, # years
    std_dev_education_init::Float64 = 3.0,
    min_education_init::Int = 0,
    max_education_init::Int = 20,
    employment_statuses_init::Vector{Symbol} = [:white_collar, :blue_collar, :underemployed, :self_employment, :unemployed],
    employment_probs_init::Vector{Float64} = [0.25, 0.25, 0.15, 0.15, 0.20],
    prob_family_abroad_init::Float64 = 0.2,
    initial_income_dist::Distribution = Normal(1500, 500), # Placeholder monthly income in a generic currency unit

    # TPB related model parameters (many of these are "inferred parameters" or "initial assumptions")
    # BA parameters
    ba_weight_econ::Float64 = 0.5,
    ba_weight_pol::Float64 = 0.5,
    ba_base_aspiration::Float64 = 0.1, # Intrinsic desire to migrate
    ba_threshold_assessment::Float64 = 0.2, # Min BA to proceed from assessment
    ba_min_val::Float64 = -2.0,
    ba_max_val::Float64 = 2.0,

    # SN parameters
    sn_base_value::Float64 = 0.1,
    sn_weight_family_abroad::Float64 = 0.3,
    sn_weight_migrated_friend::Float64 = 0.1,
    sn_societal_pressure_factor::Float64 = 0.05, # Placeholder for general societal influence
    sn_min_val::Float64 = 0.0,
    sn_max_val::Float64 = 2.0,

    # PBC parameters
    pbc_base_value::Float64 = 0.2,
    pbc_weight_education::Float64 = 0.02,
    pbc_weight_income::Float64 = 0.0001, # income units matter here
    pbc_effect_unemployed::Float64 = -0.1,
    pbc_effect_white_collar::Float64 = 0.1,
    pbc_weight_policy_complexity::Float64 = 0.2,
    pbc_weight_network_capital::Float64 = 0.05,
    pbc_min_val::Float64 = 0.0,
    pbc_max_val::Float64 = 2.0,

    # V_score parameters [cite: 187, 191, 532]
    gamma_V::Float64 = 1.0, # Scaling factor
    alpha_V::Float64 = 0.5, # Elasticity for SN
    beta_V::Float64 = 0.5,  # Elasticity for PBC
    Vh_threshold::Float64 = 0.5, # Intention threshold [cite: 192, 195]

    # Transition rate/time parameters
    lambda_av_assessment_to_intention::Float64 = 0.5, # Rate for Tav
    # Theta for Tvi (time to develop normative/control beliefs) is complex. For simplicity, use fixed probability or another exponential.
    # For MVP, let's assume a fixed probability to transition from assessment (after BA check) to intention if V >= Vh
    prob_assessment_to_intention_step::Float64 = 0.1, # If V >= Vh, this prob per step.

    # Planning and Preparation stages [cite: 203, 279] - simplified as probabilities per step
    prob_intention_to_planning_step::Float64 = 0.2,
    prob_planning_to_preparation_step::Float64 = 0.2,

    # Emigration/Dropout from preparation
    ae_emigration::Float64 = 0.10, # Base emigration rate param
    be_emigration::Float64 = 2.00, # Emigration rate scaling param
    ac_dropout::Float64 = 0.01,    # Base dropout rate param
    bc_dropout::Float64 = 2.00,    # Dropout rate scaling param

    # Network formation parameters
    range_age::Float64 = max_age_init - min_age_init,
    range_education::Float64 = Float64(max_education_init - min_education_init),
    w_age::Float64 = 3.0/6.0,
    w_ethnicity::Float64 = 2.0/6.0, # As per ODD [cite: 510]
    w_education::Float64 = 1.0/6.0,
    w_gender_for_homophily::Float64 = 1.0/6.0, # Adding gender with some weight, not in original list but is agent attribute.
    network_density_factor::Float64 = 0.05, # General scaling for network density

    # Placeholder environmental variables (can be time-series DataFrames later)
    # These are initial values, model_step! should update them
    initial_political_instability_turkey::Float64 = 0.5, # Scale 0-1
    initial_economic_conditions_turkey::Float64 = 0.3,   # Scale 0-1 (e.g. inverse unemployment or composite index)
    initial_economic_conditions_germany::Float64 = 0.7, # Scale 0-1 (e.g. job prospect index)
    initial_policy_complexity_germany::Float64 = 0.6,  # Scale 0-1
    initial_political_stability_germany::Float64 = 0.8, # Scale 0-1 (for BA calculation)

    # Placeholder German policy settings for ABC calculation
    # These would come from DEMIG/IMPIC and be time-varying
    german_policy_settings = Dict(
        :high_skill_education_threshold => 16, # years
        :base_job_offer_prob => 0.1,
        :high_skill_job_multiplier => 2.0,
        :language_skill_prob_edu_gt_12 => 0.7,
        :language_skill_prob_edu_lte_12 => 0.2,
        :financial_requirement_threshold => 2000.0 # Income units
    ),
    # Mortality Rates (placeholder: simple annual probability, refined by age/gender in agent_step!)
    base_annual_mortality_rate::Float64 = 0.005,

    # Income Accrual (placeholder: simple annual increase)
    annual_income_growth_rate::Float64 = 0.02,

    # Simulation control
    delta_t_months::Int = 1 # Each step is 1 month
)

    properties = Dict(
        :current_time_months => 0, # Simulation time in months
        :delta_t_months => delta_t_months,
        # Store parameters for easy access
        :ba_weight_econ => ba_weight_econ, :ba_weight_pol => ba_weight_pol,
        :ba_base_aspiration => ba_base_aspiration, :ba_threshold_assessment => ba_threshold_assessment,
        :ba_min_val => ba_min_val, :ba_max_val => ba_max_val,
        :sn_base_value => sn_base_value, :sn_weight_family_abroad => sn_weight_family_abroad,
        :sn_weight_migrated_friend => sn_weight_migrated_friend,
        :sn_societal_pressure_factor => sn_societal_pressure_factor,
        :sn_min_val => sn_min_val, :sn_max_val => sn_max_val,
        :pbc_base_value => pbc_base_value, :pbc_weight_education => pbc_weight_education,
        :pbc_weight_income => pbc_weight_income, :pbc_effect_unemployed => pbc_effect_unemployed,
        :pbc_effect_white_collar => pbc_effect_white_collar,
        :pbc_weight_policy_complexity => pbc_weight_policy_complexity,
        :pbc_weight_network_capital => pbc_weight_network_capital,
        :pbc_min_val => pbc_min_val, :pbc_max_val => pbc_max_val,
        :gamma_V => gamma_V, :alpha_V => alpha_V, :beta_V => beta_V, :Vh_threshold => Vh_threshold,
        :lambda_av_assessment_to_intention => lambda_av_assessment_to_intention,
        :prob_assessment_to_intention_step => prob_assessment_to_intention_step,
        :prob_intention_to_planning_step => prob_intention_to_planning_step,
        :prob_planning_to_preparation_step => prob_planning_to_preparation_step,
        :ae_emigration => ae_emigration, :be_emigration => be_emigration,
        :ac_dropout => ac_dropout, :bc_dropout => bc_dropout,
        :range_age => range_age, :range_education => range_education,
        :w_age => w_age, :w_ethnicity => w_ethnicity, :w_education => w_education,
        :w_gender_for_homophily => w_gender_for_homophily,
        :network_density_factor => network_density_factor,
        # Time-varying environmental variables (initialized, updated in model_step!)
        # These should be replaced with actual time-series data structures
        :political_instability_turkey => initial_political_instability_turkey,
        :economic_conditions_turkey => initial_economic_conditions_turkey,
        :economic_conditions_germany => initial_economic_conditions_germany,
        :policy_complexity_germany => initial_policy_complexity_germany,
        :political_stability_germany => initial_political_stability_germany, # For BA calc
        :german_policy_settings => german_policy_settings, # Can be updated over time
        :base_annual_mortality_rate => base_annual_mortality_rate,
        :annual_income_growth_rate => annual_income_growth_rate,
        # Data placeholders (e.g. for SCI if not directly used in function calls)
        # :sci_matrix => load_sci_data(), # You'd load your SCI matrix here
        # :turkstat_regional_unemployment => load_regional_unemployment_data() # DataFrame
    )

    # Create a graph with num_agents vertices upfront
    graph = SimpleGraph(num_agents) # num_agents vertices, 0 edges initially
    space = GraphSpace(graph)
    model = ABM(
        Person,
        space;
        agent_step! = agent_step!,
        model_step! = model_step!,
        properties = properties,
        scheduler = Agents.Schedulers.Randomly # Your capitalized fix
    )

    # Initialize agents and add them to specific nodes
    for i in 1:num_agents # Agent 'i' will be placed on node 'i'
       # Generate agent attributes
       age = rand(Truncated(Normal(mean_age_init, std_dev_age_init), min_age_init, max_age_init)) # Float64
       gender = rand() < prob_male_init ? :male : :female # Symbol
       province = rand(1:num_provinces) # Int
       education = rand(min_education_init:max_education_init) # Int
       employment_status_idx = rand(Distributions.Categorical(employment_probs_init / sum(employment_probs_init)))
       mapped_employment_status = employment_statuses_init[employment_status_idx] # Symbol
       income = max(500.0, rand(initial_income_dist)) # Float64
       has_family_abroad = rand() < prob_family_abroad_init # Bool
       current_skill_level = get_skill_level(education, mapped_employment_status) # Symbol
       age_of_interest_val = generate_age_of_interest(current_skill_level) # Float64

       agent_properties_tuple = (
           age,                          # 1. Float64
           gender,                       # 2. Symbol
           province,                     # 3. Int
           education,                    # 4. Int
           mapped_employment_status,     # 5. Symbol
           income,                       # 6. Float64
           has_family_abroad,            # 7. Bool
           :never_consider,              # 8. Symbol (migration_stage)
           0.0,                          # 9. Float64 (time_in_current_stage)
           0.0,                          # 10. Float64 (behavioral_attitude)
           0.0,                          # 11. Float64 (subjective_norm)
           0.0,                          # 12. Float64 (perceived_behavioral_control)
           0.0,                          # 13. Float64 (actual_behavioral_control)
           0.0,                          # 14. Float64 (V_score)
           model.political_instability_turkey, # 15. Float64
           model.economic_conditions_turkey,   # 16. Float64
           model.economic_conditions_germany,  # 17. Float64
           model.policy_complexity_germany,    # 18. Float64
           current_skill_level,             # 19. Symbol
           age_of_interest_val              # 20. Float64
       )
       # !!! REPLACEMENT FOR THE add_agent! CALL !!!
    # Pass all 20 custom properties directly and in the correct order:
        add_agent!(
            i,            # node_id for GraphSpace
            model,
            age,                          # 1st custom property (Float64)
            gender,                       # 2nd (Symbol)
            province,                     # 3rd (Int)
            education,                    # 4th (Int)
            mapped_employment_status,     # 5th (Symbol)
            income,                       # 6th (Float64)
            has_family_abroad,            # 7th (Bool)
            :never_consider,              # 8th (Symbol) - initial migration_stage
            0.0,                          # 9th (Float64) - initial time_in_current_stage
            0.0,                          # 10th (Float64) - initial behavioral_attitude
            0.0,                          # 11th (Float64) - initial subjective_norm
            0.0,                          # 12th (Float64) - initial perceived_behavioral_control
            0.0,                          # 13th (Float64) - initial actual_behavioral_control
            0.0,                          # 14th (Float64) - initial V_score
            model.political_instability_turkey, # 15th - initial sensed value
            model.economic_conditions_turkey,   # 16th - initial sensed value
            model.economic_conditions_germany,  # 17th - initial sensed value
            model.policy_complexity_germany,    # 18th - initial sensed value
            current_skill_level,             # 19th (Symbol)
            age_of_interest_val              # 20th (Float64)
        )
    end

    # Form initial network AFTER all agents are added and assigned to nodes
    form_initial_network!(model)

    println("Model initialized with $num_agents agents.")
    return model
end


# --- Agent Step Function ---
function agent_step!(agent::Person, model::ABM)
    # 0. Agent aging and life course events
    agent.age += model.delta_t_months / 12.0
    agent.time_in_current_stage += model.delta_t_months

    # Income accrual (simplified annual update if current month is start of year)
    if model.current_time_months % 12 == 0 && model.current_time_months > 0
        agent.income *= (1 + model.annual_income_growth_rate)
    end

    # Mortality
    # Convert annual rate to monthly
    monthly_mortality_rate = 1 - (1 - model.base_annual_mortality_rate)^(model.delta_t_months / 12.0)
    # Adjust for age (very simplified, proper life tables needed for accuracy)
    if agent.age > 60
        monthly_mortality_rate *= (1 + (agent.age - 60) * 0.05) # Increase rate for older agents
    end
    if rand() < monthly_mortality_rate
        kill_agent!(agent, model)
        return # Agent is removed
    end
    
    # If agent has already migrated or decided to stay, no further action
    if agent.migration_stage == :migrated || agent.migration_stage == :stay
        return
    end

    # 1. Sensing: Update perceived environment from model properties
    # Assume a simple lag: perceptions are based on the previous state of the model's environment
    # For this discrete step, this is implicitly handled if model_step! updates env vars before agent_step! for the *next* step.
    # Or, explicitly use lagged values if stored. For now, use current model values.
    agent.perceived_political_instability_turkey = model.political_instability_turkey
    agent.perceived_economic_conditions_turkey = model.economic_conditions_turkey # TODO: province-specific sensing [cite: 94]
    agent.perceived_economic_conditions_germany = model.economic_conditions_germany
    agent.perceived_policy_complexity = model.policy_complexity_germany

    # 2. Update TPB components based on current perceptions and network
    current_BA = calculate_BA(agent, model)
    agent.behavioral_attitude = current_BA # Update agent's BA

    current_SN = calculate_SN(agent, model) # SN depends on network state
    agent.subjective_norm = current_SN

    current_PBC = calculate_PBC(agent, model) # PBC also depends on network and self-attributes
    agent.perceived_behavioral_control = current_PBC
    
    # 3. Migration Stage Transitions

    # --- :never_consider to :assessment ---
    if agent.migration_stage == :never_consider
        # Transition based on age_of_interest
        if agent.age >= agent.age_of_interest
            agent.migration_stage = :assessment
            agent.time_in_current_stage = 0.0
            println("Agent $(agent.id) moved to :assessment at age $(agent.age)")
        end

    # --- :assessment to :intention (or :stay) ---
    elseif agent.migration_stage == :assessment
        # BA check [cite: 275]
        if current_BA < model.ba_threshold_assessment
            agent.migration_stage = :stay
            agent.time_in_current_stage = 0.0
            println("Agent $(agent.id) dropped to :stay from :assessment (low BA)")
            return
        end

        # Calculate V_score [cite: 187, 532]
        agent.V_score = calculate_V_score(current_SN, current_PBC, model)

        # Time in stage Tav, Tvi logic[cite: 529, 533]. Approximated by prob_assessment_to_intention_step
        # ODD mentions Tav = -lambda_av * ln(u) and Tvi based on V_score distance from median.
        # Simplified: fixed probability per step if V is high enough, after a minimum time.
        # Or, more aligned with ODD: prob_transition = 1 - exp(-model.lambda_av_assessment_to_intention * model.delta_t_months)
        
        # Simplified transition logic for assessment:
        # Assume this probability captures the combined waiting times Tav and Tvi implicitly.
        # This part needs careful mapping from continuous time rates to discrete probabilities.
        # A common way for rate lambda: P(event in dt) = 1 - exp(-lambda*dt)
        # For Tav (attitude formation time):
        prob_attitude_consolidated_this_step = 1 - exp(-model.lambda_av_assessment_to_intention * model.delta_t_months)

        if rand() < prob_attitude_consolidated_this_step # Simulates passing Tav
            if agent.V_score >= model.Vh_threshold
                agent.migration_stage = :intention
                agent.time_in_current_stage = 0.0
                println("Agent $(agent.id) moved to :intention (V_score $(agent.V_score) >= $(model.Vh_threshold))")
            else
                # Check if enough time has passed in assessment to decide to drop out due to low V
                # The ODD implies Vh check can lead to dropout [cite: 194]
                # Let's assume if attitude is consolidated but V is low, dropout.
                agent.migration_stage = :stay
                agent.time_in_current_stage = 0.0
                println("Agent $(agent.id) dropped to :stay from :assessment (low V_score: $(agent.V_score))")
            end
        end

    # --- :intention to :planning (or :stay) ---
    elseif agent.migration_stage == :intention
        # Re-evaluation of intention
        if agent.V_score < model.Vh_threshold || current_BA < model.ba_threshold_assessment # If conditions worsened
            agent.migration_stage = :stay
            agent.time_in_current_stage = 0.0
            println("Agent $(agent.id) dropped to :stay from :intention (reevaluation failed)")
            return
        end
        # Stochastic waiting time (exponential dist) [cite: 203, 279] -> probability per step
        if rand() < model.prob_intention_to_planning_step # This probability should be derived from a rate
            agent.migration_stage = :planning
            agent.time_in_current_stage = 0.0
            println("Agent $(agent.id) moved to :planning")
        end

    # --- :planning to :preparation (or :stay) ---
    elseif agent.migration_stage == :planning
        # Re-evaluation
        if agent.V_score < model.Vh_threshold || current_BA < model.ba_threshold_assessment
            agent.migration_stage = :stay
            agent.time_in_current_stage = 0.0
            println("Agent $(agent.id) dropped to :stay from :planning (reevaluation failed)")
            return
        end
        if rand() < model.prob_planning_to_preparation_step # This probability should be derived from a rate
            agent.migration_stage = :preparation
            agent.time_in_current_stage = 0.0
            println("Agent $(agent.id) moved to :preparation")
        end

    # --- :preparation to :migrated or :stay (Emigration Attempt) ---
    elseif agent.migration_stage == :preparation
        # Re-evaluation (optional here, but good practice, or assume commitment is high)
        if agent.V_score < model.Vh_threshold || current_BA < model.ba_threshold_assessment
            agent.migration_stage = :stay
            agent.time_in_current_stage = 0.0
            println("Agent $(agent.id) dropped to :stay from :preparation (last reevaluation failed)")
            return
        end

        # Calculate ABC
        agent.actual_behavioral_control = calculate_ABC(agent, model)
        abc_val = agent.actual_behavioral_control

        # Emigration and Dropout Rates
        mu_ie = model.ae_emigration * exp(model.be_emigration * abc_val)
        mu_ic = model.ac_dropout * exp(model.bc_dropout * (1.0 - abc_val))
        
        mu_total_exit = mu_ie + mu_ic
        if mu_total_exit == 0.0 # Avoid division by zero; means no chance of exit this step under these params
            prob_exit_this_step = 0.0
            prob_emigrate_given_exit = 0.0
        else
            # Probability of exiting the :preparation stage this step
            prob_exit_this_step = 1.0 - exp(-mu_total_exit * model.delta_t_months)
            prob_emigrate_given_exit = mu_ie / mu_total_exit
        end
        
        if rand() < prob_exit_this_step # Agent exits the preparation stage
            if rand() < prob_emigrate_given_exit # Bernoulli trial
                agent.migration_stage = :migrated
                println("Agent $(agent.id) MIGRATED! (ABC: $abc_val)")
            else
                agent.migration_stage = :stay
                println("Agent $(agent.id) dropped to :stay from :preparation (failed attempt/dropout, ABC: $abc_val)")
            end
            agent.time_in_current_stage = 0.0
        end
    end
end

# --- Model Step Function ---
function model_step!(model::ABM)
    model.current_time_months += model.delta_t_months
    println("\n--- Model Step: Month $(model.current_time_months) ---")

    # Update exogenous environmental variables (placeholder: simple random walk or load from series)
    #
    # Example: model.economic_conditions_turkey = update_econ_turkey(model.current_time_months, model.base_econ_turkey_series)
    # For now, simple fluctuations as placeholders:
    model.political_instability_turkey = clamp(model.political_instability_turkey + rand(Normal(0, 0.01)), 0.1, 0.9)
    model.economic_conditions_turkey = clamp(model.economic_conditions_turkey + rand(Normal(0, 0.02)), 0.1, 0.9)
    model.economic_conditions_germany = clamp(model.economic_conditions_germany + rand(Normal(0, 0.01)), 0.2, 0.9)
    model.policy_complexity_germany = clamp(model.policy_complexity_germany + rand(Normal(0, 0.01)), 0.2, 0.9)
    
    # Potentially update german_policy_settings if policies change over time
    # e.g., if model.current_time_months corresponds to a known policy change date.
    # if model.current_time_months == 24 # Example: Policy change at month 24
    #   model.german_policy_settings[:high_skill_education_threshold] = 17
    #   println("Policy change: German high-skill education threshold increased.")
    # end

    # --- Data Collection (Example) ---
    # This can be done more systematically using Agents.jl's `run!` function with `adata` and `mdata`
    # For manual tracking within model_step:
    num_migrated_total = count(a -> a.migration_stage == :migrated, allagents(model))
    num_assessment = count(a -> a.migration_stage == :assessment, allagents(model))
    num_intention = count(a -> a.migration_stage == :intention, allagents(model))
    # ... and so on for other stages and metrics

    println("Month $(model.current_time_months): Migrated Agents = $num_migrated_total, In Assessment = $num_assessment, In Intention = $num_intention")

    # For more detailed data logging at intervals:
    # if model.current_time_months % model.reporting_interval == 0
    #   # Log detailed agent data and aggregate model data
    # end
end


# --- Running the Simulation ---
function run_simulation()
    println("Starting Turkey-Germany Migration ABM...")
    
    # Initialize the model
    # You can change num_agents and other parameters here
    model = initialize_model(num_agents = 100) # Small number for quick test

    # Define data to collect
    # Agent data: collect for agents in specific stages or all
    agent_data_to_collect = [:id, :age, :gender, :education, :employment_status, :province,
                             :migration_stage, :time_in_current_stage,
                             :behavioral_attitude, :subjective_norm, :perceived_behavioral_control,
                             :actual_behavioral_control, :V_score, :has_family_abroad]
    
    # Model data: collect at each step
    model_data_to_collect = [:current_time_months,
                             # Example of a function to compute aggregate data
                             (m -> count(a.migration_stage == :migrated for a in allagents(m))),
                             (m -> count(a.migration_stage == :assessment for a in allagents(m))),
                             (m -> count(a.migration_stage == :intention for a in allagents(m))),
                             (m -> count(a.migration_stage == :planning for a in allagents(m))),
                             (m -> count(a.migration_stage == :preparation for a in allagents(m))),
                             (m -> count(a.migration_stage == :stay for a in allagents(m))),
                             :economic_conditions_turkey, :economic_conditions_germany,
                             :political_instability_turkey, :policy_complexity_germany
                            ]
    
    # Number of steps (simulation runs for 84 months [cite: 102])
    num_steps = 84 

    # Run the simulation
    # `when` controls how often data is collected. `1:num_steps` collects at every step.
    # `adata_scheduler` and `mdata_scheduler` could be used for more complex scheduling.
    agent_df, model_df = run!(model, agent_step!, model_step!, num_steps;
                              adata = agent_data_to_collect,
                              mdata = model_data_to_collect,
                              when = 1:num_steps)

    println("\nSimulation Finished.")
    println("Collected Agent Data:")
    println(first(agent_df, 5)) # Print first 5 rows of agent data
    println("\nCollected Model Data:")
    println(first(model_df, 5))  # Print first 5 rows of model data

    # You can now save agent_df and model_df to CSV files or perform analysis
    # using CSV, DataFrames
    # using CSV
    # CSV.write("agent_data.csv", agent_df)
    # CSV.write("model_data.csv", model_df)
    # println("\nData saved to agent_data.csv and model_data.csv")

    return agent_df, model_df
end

# To run the simulation:
# agent_results, model_results = run_simulation();

# --- Notes for User ---
# 1.  **Data Placeholders**: Many functions like `calculate_BA`, `calculate_SN`, `calculate_PBC`, `calculate_ABC`,
#     and parts of `initialize_model` use placeholder logic or simplified distributions. You MUST replace these
#     with actual data loading (e.g., from CSVs for TurkStat, EUMAGINE, TRANSMIT, SCI data, policy time series)
#     and more nuanced calculations as per your full ODD+D and empirical sources.
# 2.  **Parameter Calibration**: Many parameters (e.g., `gamma_V`, `Vh_threshold`, `ae_emigration`, etc.) are
#     flagged as "inferred parameters" or "initial assumptions" in the ODD+D. The current values are
#     placeholders or the initial assumptions from the ODD. These need to be calibrated, potentially using
#     Approximate Bayesian Computation (ABC) as suggested.
# 3.  **Continuous Time vs. Discrete Time**: This model uses discrete monthly time steps. If true event-based
#     scheduling is critical, `Agents.jl` can be adapted, but it's more complex. The current approach with
#     probabilities derived from rates is a common simplification.
# 4.  **Network Implementation**: The `form_initial_network!` function provides a basic structure for network
#     creation based on homophily and provincial connectivity. Ensure your SCI data and normalization are correct.
#     The ODD details specific weights for demographic similarity.
# 5.  **Sensing Lags**: The ODD mentions lagged sensing of environmental variables. This is currently
#     implicit if `model_step!` updates the environment before `agent_step!` for the *next* period. For more
#     explicit lags, you'd need to store historical environmental data in the model.
# 6.  **Province-Specific Sensing**: The ODD indicates agents sense some conditions at their province level. [cite: 94, 305]
#     The `calculate_BA` (for Turkish conditions) should incorporate this, e.g., by accessing province-specific
#     unemployment from a model property.
# 7.  **Re-evaluation Logic**: The re-evaluation of intention at each stage transition is implemented. Ensure the
#     conditions for dropping out (e.g., "intention has significantly weakened") are defined as per your framework.
# 8.  **ABC Complexity**: `calculate_ABC` is highly simplified. A realistic implementation requires detailed
#     mapping of German visa rules.
# 9.  **"Inferred Parameters for ABC"**: These are crucial for model behavior and require calibration.
#     The ODD names several: gamma, alpha, beta for V_score; Vh_threshold; lambda_av; theta for Tvi;
#     ae, be, ac, bc for emigration/dropout. [cite: 191, 195, 534, 536, 546]

println("Julia code for Turkey-Germany Migration ABM is ready to be run and adapted.")
println("Remember to replace placeholder data and calibrate parameters.")

agent_results, model_results = run_simulation();