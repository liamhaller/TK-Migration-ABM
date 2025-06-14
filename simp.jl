# Necessary Julia packages
# You might need to install them first using Pkg.add("PackageName")
using Agents
using Random
using Distributions
using DataFrames
using Graphs
using CSV

# --- Agent Definition (Updated Syntax) ---
@agent struct Person(GraphAgent)
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
function get_skill_level(education::Int, employment_status::Symbol)::Symbol
    if education >= 16 && (employment_status == :white_collar || employment_status == :self_employment)
        return :high_skill
    else
        return :low_medium_skill
    end
end

function generate_age_of_interest(skill_level::Symbol, min_age_consideration::Float64 = 15.0)::Float64
    if skill_level == :high_skill
        dist = Truncated(Normal(20, 2), min_age_consideration, Inf)
    else
        dist = Truncated(Normal(19, 4), min_age_consideration, Inf)
    end
    return rand(dist)
end

function calculate_BA(agent::Person, model::ABM)::Float64
    econ_diff_eval = (agent.perceived_economic_conditions_germany - agent.perceived_economic_conditions_turkey) * model.ba_weight_econ
    pol_diff_eval = (model.political_stability_germany - agent.perceived_political_instability_turkey) * model.ba_weight_pol
    base_aspiration = model.ba_base_aspiration
    ba_score = base_aspiration + econ_diff_eval + pol_diff_eval
    return clamp(ba_score, model.ba_min_val, model.ba_max_val)
end

function calculate_SN(agent::Person, model::ABM)::Float64
    sn_score = model.sn_base_value
    num_migrated_friends = 0
    # Using getfield for robust space access
    current_graph = getfield(model, :space).graph
    if Graphs.nv(current_graph) > 0 && !isempty(nearby_ids(agent, model))
        for neighbor_id in nearby_ids(agent, model)
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
    sn_score += model.sn_societal_pressure_factor
    return clamp(sn_score, model.sn_min_val, model.sn_max_val)
end

function calculate_PBC(agent::Person, model::ABM)::Float64
    pbc_score = model.pbc_base_value
    pbc_score += agent.education * model.pbc_weight_education
    pbc_score += agent.income * model.pbc_weight_income
    if agent.employment_status == :unemployed
        pbc_score += model.pbc_effect_unemployed
    elseif agent.employment_status == :white_collar
        pbc_score += model.pbc_effect_white_collar
    end
    pbc_score -= agent.perceived_policy_complexity * model.pbc_weight_policy_complexity
    known_migrants_in_network = 0
    # Using getfield for robust space access
    current_graph = getfield(model, :space).graph
    if Graphs.nv(current_graph) > 0 && !isempty(nearby_ids(agent, model))
        for neighbor_id in nearby_ids(agent, model)
            neighbor = model[neighbor_id]
            if neighbor.migration_stage == :migrated || neighbor.has_family_abroad
                known_migrants_in_network +=1
            end
        end
    end
    pbc_score += known_migrants_in_network * model.pbc_weight_network_capital
    return clamp(pbc_score, model.pbc_min_val, model.pbc_max_val)
end

function calculate_V_score(sn::Float64, pbc::Float64, model::ABM)::Float64
    safe_sn = max(0.01, sn)
    safe_pbc = max(0.01, pbc)
    return model.gamma_V * (safe_sn^model.alpha_V) * (safe_pbc^model.beta_V)
end

function calculate_ABC(agent::Person, model::ABM)::Float64
    education_match = 0.0
    required_edu_high_skill = model.german_policy_settings[:high_skill_education_threshold]
    if agent.skill_level == :high_skill && agent.education >= required_edu_high_skill
        education_match = 0.4
    elseif agent.skill_level == :low_medium_skill
        education_match = 0.2
    end
    prob_job_offer = model.german_policy_settings[:base_job_offer_prob]
    if agent.skill_level == :high_skill
        prob_job_offer *= model.german_policy_settings[:high_skill_job_multiplier]
    end
    job_offer_points = rand() < prob_job_offer ? 0.3 : 0.0
    language_points = 0.0
    if agent.education > 12
        language_points = rand() < model.german_policy_settings[:language_skill_prob_edu_gt_12] ? 0.15 : 0.0
    else
        language_points = rand() < model.german_policy_settings[:language_skill_prob_edu_lte_12] ? 0.05 : 0.0
    end
    financial_points = 0.0
    if agent.income >= model.german_policy_settings[:financial_requirement_threshold]
        financial_points = 0.15
    end
    abc_score = clamp(education_match + job_offer_points + language_points + financial_points, 0.0, 1.0)
    return abc_score
end

function demographic_similarity(agent1::Person, agent2::Person, model::ABM)::Float64
    s_age = 1.0 - (abs(agent1.age - agent2.age) / model.range_age)
    s_gender = agent1.gender == agent2.gender ? 1.0 : 0.0
    s_education = 1.0 - (abs(agent1.education - agent2.education) / model.range_education)
    total_similarity = (model.w_age * s_age) +
                       (model.w_gender_for_homophily * s_gender) +
                       (model.w_education * s_education) +
                       (model.w_ethnicity * 1.0)
    return total_similarity / (model.w_age + model.w_gender_for_homophily + model.w_education + model.w_ethnicity)
end

function provincial_connectivity_factor(province1::Int, province2::Int, model::ABM)::Float64
    if province1 == province2
        return 1.0
    else
        raw_sci = 1.0 / (1.0 + abs(province1 - province2)) # Placeholder
        return raw_sci / 1.0 # Placeholder normalization
    end
end

function form_initial_network!(model::ABM)
    println("Forming initial social network...")
    agents_vector = collect(allagents(model))
    num_actual_agents = length(agents_vector)
    # Using getfield for robust space access
    current_graph = getfield(model, :space).graph

    for i in 1:num_actual_agents
        for j in (i+1):num_actual_agents
            agent1 = agents_vector[i]
            agent2 = agents_vector[j]
            sim = demographic_similarity(agent1, agent2, model)
            prov_conn = provincial_connectivity_factor(agent1.province, agent2.province, model)
            connection_prob = sim * prov_conn * model.network_density_factor
            if rand() < connection_prob
                Graphs.add_edge!(current_graph, agent1.pos, agent2.pos)
            end
        end
    end
    println("Network formation complete. Number of edges: ", Graphs.ne(current_graph))
end

# --- Model Initialization ---
function initialize_model(;
    num_agents::Int = 100, # Reduced for faster testing initially
    mean_age_init::Float64 = 35.0, std_dev_age_init::Float64 = 10.0,
    min_age_init::Float64 = 18.0, max_age_init::Float64 = 65.0,
    prob_male_init::Float64 = 0.5, num_provinces::Int = 81,
    mean_education_init::Float64 = 10.0, std_dev_education_init::Float64 = 3.0,
    min_education_init::Int = 0, max_education_init::Int = 20,
    employment_statuses_init::Vector{Symbol} = [:white_collar, :blue_collar, :underemployed, :self_employment, :unemployed],
    employment_probs_init::Vector{Float64} = [0.25, 0.25, 0.15, 0.15, 0.20],
    prob_family_abroad_init::Float64 = 0.2,
    initial_income_dist::Distribution = Normal(1500, 500),
    ba_weight_econ::Float64 = 0.5, ba_weight_pol::Float64 = 0.5,
    ba_base_aspiration::Float64 = 0.1, ba_threshold_assessment::Float64 = 0.2,
    ba_min_val::Float64 = -2.0, ba_max_val::Float64 = 2.0,
    sn_base_value::Float64 = 0.1, sn_weight_family_abroad::Float64 = 0.3,
    sn_weight_migrated_friend::Float64 = 0.1, sn_societal_pressure_factor::Float64 = 0.05,
    sn_min_val::Float64 = 0.0, sn_max_val::Float64 = 2.0,
    pbc_base_value::Float64 = 0.2, pbc_weight_education::Float64 = 0.02,
    pbc_weight_income::Float64 = 0.0001, pbc_effect_unemployed::Float64 = -0.1,
    pbc_effect_white_collar::Float64 = 0.1, pbc_weight_policy_complexity::Float64 = 0.2,
    pbc_weight_network_capital::Float64 = 0.05,
    pbc_min_val::Float64 = 0.0, pbc_max_val::Float64 = 2.0,
    gamma_V::Float64 = 1.0, alpha_V::Float64 = 0.5, beta_V::Float64 = 0.5,
    Vh_threshold::Float64 = 0.5,
    lambda_av_assessment_to_intention::Float64 = 0.5,
    prob_assessment_to_intention_step::Float64 = 0.1,
    prob_intention_to_planning_step::Float64 = 0.2,
    prob_planning_to_preparation_step::Float64 = 0.2,
    ae_emigration::Float64 = 0.10, be_emigration::Float64 = 2.00,
    ac_dropout::Float64 = 0.01, bc_dropout::Float64 = 2.00,
    range_age::Float64 = max_age_init - min_age_init,
    range_education::Float64 = Float64(max_education_init - min_education_init),
    w_age::Float64 = 3.0/6.0, w_ethnicity::Float64 = 2.0/6.0,
    w_education::Float64 = 1.0/6.0, w_gender_for_homophily::Float64 = 1.0/6.0,
    network_density_factor::Float64 = 0.05,
    initial_political_instability_turkey::Float64 = 0.5,
    initial_economic_conditions_turkey::Float64 = 0.3,
    initial_economic_conditions_germany::Float64 = 0.7,
    initial_policy_complexity_germany::Float64 = 0.6,
    initial_political_stability_germany::Float64 = 0.8,
    german_policy_settings = Dict(
        :high_skill_education_threshold => 16,
        :base_job_offer_prob => 0.1,
        :high_skill_job_multiplier => 2.0,
        :language_skill_prob_edu_gt_12 => 0.7,
        :language_skill_prob_edu_lte_12 => 0.2,
        :financial_requirement_threshold => 2000.0
    ),
    base_annual_mortality_rate::Float64 = 0.005,
    annual_income_growth_rate::Float64 = 0.02,
    delta_t_months::Int = 1
)
    properties = Dict(
        :current_time_months => 0, :delta_t_months => delta_t_months,
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
        :political_instability_turkey => initial_political_instability_turkey,
        :economic_conditions_turkey => initial_economic_conditions_turkey,
        :economic_conditions_germany => initial_economic_conditions_germany,
        :policy_complexity_germany => initial_policy_complexity_germany,
        :political_stability_germany => initial_political_stability_germany,
        :german_policy_settings => german_policy_settings,
        :base_annual_mortality_rate => base_annual_mortality_rate,
        :annual_income_growth_rate => annual_income_growth_rate,
    )

    graph = SimpleGraph(num_agents)
    space = GraphSpace(graph)
    # Corrected ABM constructor with stepping functions and correct scheduler name
    model = ABM(
        Person, space;
        agent_step! = agent_step!, model_step! = model_step!,
        properties = properties, scheduler = Agents.Schedulers.fastest # Capital 'R'
    )

    for i in 1:num_agents
       age_val = rand(Truncated(Normal(mean_age_init, std_dev_age_init), min_age_init, max_age_init))
       gender_val = rand() < prob_male_init ? :male : :female
       province_val = rand(1:num_provinces)
       education_val = rand(min_education_init:max_education_init)
       employment_status_idx = rand(Distributions.Categorical(employment_probs_init ./ sum(employment_probs_init)))
       mapped_employment_status_val = employment_statuses_init[employment_status_idx]
       income_val = max(500.0, rand(initial_income_dist))
       has_family_abroad_val = rand() < prob_family_abroad_init
       current_skill_level_val = get_skill_level(education_val, mapped_employment_status_val)
       age_of_interest_val_val = generate_age_of_interest(current_skill_level_val) # Renamed to avoid conflict

        # Using direct property listing for add_agent!
        add_agent!(
            i, model,
            age_val, gender_val, province_val, education_val, mapped_employment_status_val, income_val,
            has_family_abroad_val,
            :never_consider, 0.0, 
            0.0, 0.0, 0.0,       
            0.0,                   
            0.0,                   
            model.political_instability_turkey, 
            model.economic_conditions_turkey,
            model.economic_conditions_germany,
            model.policy_complexity_germany,
            current_skill_level_val,
            age_of_interest_val_val 
        )
    end

    form_initial_network!(model) # Call after agents are added
    println("Model initialized with $num_agents agents.")
    return model
end

# --- Agent Step Function ---
function agent_step!(agent::Person, model::ABM)
    agent.age += model.delta_t_months / 12.0
    agent.time_in_current_stage += model.delta_t_months

    if model.current_time_months % 12 == 0 && model.current_time_months > 0
        agent.income *= (1 + model.annual_income_growth_rate)
    end

 #   monthly_mortality_rate = 1 - (1 - model.base_annual_mortality_rate)^(model.delta_t_months / 12.0)
 #   if agent.age > 60
 #       monthly_mortality_rate *= (1 + (agent.age - 60) * 0.05)
 #   end
 #   if rand() < monthly_mortality_rate
 #       kill_agent!(agent, model)
 #       return
 #   end
    
    if agent.migration_stage == :migrated || agent.migration_stage == :stay
        return
    end

    agent.perceived_political_instability_turkey = model.political_instability_turkey
    agent.perceived_economic_conditions_turkey = model.economic_conditions_turkey
    agent.perceived_economic_conditions_germany = model.economic_conditions_germany
    agent.perceived_policy_complexity = model.policy_complexity_germany

    current_BA = calculate_BA(agent, model)
    agent.behavioral_attitude = current_BA
    current_SN = calculate_SN(agent, model)
    agent.subjective_norm = current_SN
    current_PBC = calculate_PBC(agent, model)
    agent.perceived_behavioral_control = current_PBC
    
    if agent.migration_stage == :never_consider
        if agent.age >= agent.age_of_interest
            agent.migration_stage = :assessment
            agent.time_in_current_stage = 0.0
            # println("Agent $(agent.id) moved to :assessment at age $(agent.age)")
        end
    elseif agent.migration_stage == :assessment
        if current_BA < model.ba_threshold_assessment
            agent.migration_stage = :stay
            agent.time_in_current_stage = 0.0
            # println("Agent $(agent.id) dropped to :stay from :assessment (low BA)")
            return
        end
        agent.V_score = calculate_V_score(current_SN, current_PBC, model)
        prob_attitude_consolidated_this_step = 1 - exp(-model.lambda_av_assessment_to_intention * model.delta_t_months)
        if rand() < prob_attitude_consolidated_this_step
            if agent.V_score >= model.Vh_threshold
                agent.migration_stage = :intention
                agent.time_in_current_stage = 0.0
                # println("Agent $(agent.id) moved to :intention (V_score $(agent.V_score) >= $(model.Vh_threshold))")
            else
                agent.migration_stage = :stay
                agent.time_in_current_stage = 0.0
                # println("Agent $(agent.id) dropped to :stay from :assessment (low V_score: $(agent.V_score))")
            end
        end
    elseif agent.migration_stage == :intention
        if agent.V_score < model.Vh_threshold || current_BA < model.ba_threshold_assessment
            agent.migration_stage = :stay
            agent.time_in_current_stage = 0.0
            # println("Agent $(agent.id) dropped to :stay from :intention (reevaluation failed)")
            return
        end
        if rand() < model.prob_intention_to_planning_step
            agent.migration_stage = :planning
            agent.time_in_current_stage = 0.0
            # println("Agent $(agent.id) moved to :planning")
        end
    elseif agent.migration_stage == :planning
        if agent.V_score < model.Vh_threshold || current_BA < model.ba_threshold_assessment
            agent.migration_stage = :stay
            agent.time_in_current_stage = 0.0
            # println("Agent $(agent.id) dropped to :stay from :planning (reevaluation failed)")
            return
        end
        if rand() < model.prob_planning_to_preparation_step
            agent.migration_stage = :preparation
            agent.time_in_current_stage = 0.0
            # println("Agent $(agent.id) moved to :preparation")
        end
    elseif agent.migration_stage == :preparation
        if agent.V_score < model.Vh_threshold || current_BA < model.ba_threshold_assessment
            agent.migration_stage = :stay
            agent.time_in_current_stage = 0.0
            # println("Agent $(agent.id) dropped to :stay from :preparation (last reevaluation failed)")
            return
        end
        agent.actual_behavioral_control = calculate_ABC(agent, model)
        abc_val = agent.actual_behavioral_control
        mu_ie = model.ae_emigration * exp(model.be_emigration * abc_val)
        mu_ic = model.ac_dropout * exp(model.bc_dropout * (1.0 - abc_val))
        mu_total_exit = mu_ie + mu_ic
        prob_exit_this_step = (mu_total_exit == 0.0) ? 0.0 : (1.0 - exp(-mu_total_exit * model.delta_t_months))
        prob_emigrate_given_exit = (mu_total_exit == 0.0) ? 0.0 : (mu_ie / mu_total_exit)
        
        if rand() < prob_exit_this_step
            if rand() < prob_emigrate_given_exit
                agent.migration_stage = :migrated
                # println("Agent $(agent.id) MIGRATED! (ABC: $abc_val)")
            else
                agent.migration_stage = :stay
                # println("Agent $(agent.id) dropped to :stay from :preparation (failed/dropout, ABC: $abc_val)")
            end
            agent.time_in_current_stage = 0.0
        end
    end
end

# --- Model Step Function ---
function model_step!(model::ABM)
    model.current_time_months += model.delta_t_months
    # Update environmental variables
    model.political_instability_turkey = clamp(model.political_instability_turkey + rand(Normal(0, 0.01)), 0.1, 0.9)
    model.economic_conditions_turkey = clamp(model.economic_conditions_turkey + rand(Normal(0, 0.02)), 0.1, 0.9)
    model.economic_conditions_germany = clamp(model.economic_conditions_germany + rand(Normal(0, 0.01)), 0.2, 0.9)
    model.policy_complexity_germany = clamp(model.policy_complexity_germany + rand(Normal(0, 0.01)), 0.2, 0.9)
    
    # Optional: For less verbose output during full runs, comment out the detailed step logging
    # if model.current_time_months % 12 == 0 # Log once a year or less frequently
    #     num_migrated_total = count(a -> a.migration_stage == :migrated, allagents(model))
    #     println("--- Model Step: Month $(model.current_time_months) --- Migrated: $num_migrated_total")
    # end
end

# --- Running the Simulation ---
function run_simulation()
    println("Starting Turkey-Germany Migration ABM...")
    model = initialize_model(num_agents = 100) # Start with a small number for testing
    
    agent_data_to_collect = [:age, :gender, :education, :employment_status, :province,
                             :migration_stage, :time_in_current_stage,
                             :behavioral_attitude, :subjective_norm, :perceived_behavioral_control,
                             :actual_behavioral_control, :V_score, :has_family_abroad]
    
    model_data_to_collect = [:current_time_months,
                             (m -> count(a.migration_stage == :migrated for a in allagents(m))),
                             (m -> count(a.migration_stage == :assessment for a in allagents(m))),
                             (m -> count(a.migration_stage == :intention for a in allagents(m))),
                             (m -> count(a.migration_stage == :planning for a in allagents(m))),
                             (m -> count(a.migration_stage == :preparation for a in allagents(m))),
                             (m -> count(a.migration_stage == :stay for a in allagents(m))),
                             :economic_conditions_turkey, :economic_conditions_germany,
                             :political_instability_turkey, :policy_complexity_germany]
    
    num_steps = 84
    agent_df, model_df = run!(model, num_steps; # agent_step! and model_step! are removed
                          adata = agent_data_to_collect,
                          mdata = model_data_to_collect,
                          when = 1:num_steps)

    println("\nSimulation Finished.")
    # Optionally print more data or save to CSV
    # println("Collected Agent Data (first 5 rows):"); println(first(agent_df, 5))
    #println("\nCollected Model Data (first 5 rows):"); println(first(model_df, 5))
    # using CSV
     CSV.write("agent_data_full.csv", agent_df)
     CSV.write("model_data_full.csv", model_df)
     println("\nData saved to CSV files.")
    return agent_df, model_df
end

println("Julia code for Turkey-Germany Migration ABM (Full Version) is ready.")
println("Remember to replace placeholder data and calibrate parameters.")
println("To run: call run_simulation() from the REPL after executing this file.")

# To run the simulation (type this in REPL after executing the file):
agent_results, model_results = run_simulation();