#!/home/dwinters/ascend-lab/publications/tms-meta/code/env/venv_tms-meta/bin/python

## tms meta-analysis
# Drew E. Winters, PhD.

#### ----- Setup ----- ####
# Environment-------------------------------------------------------------------------
# - run as a script and the above shebang line already runs in that environment


# Packages----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import hiremeta as hm

## suppressing unnecessary warning_________________
np.seterr(over="ignore")
##- This warning occurs when pymc sampling temporarily samples extreme values to adjust its trajectory.
##- Investigation on this dataset found this to be harmless for results so we are suppressing it.


# Filepath Setup----------------------------------------------------------------------
figures_base = "/home/dwinters/ascend-lab/publications/tms-meta/code/90_figures"
result_table_base = "/home/dwinters/ascend-lab/publications/tms-meta/code/90_tables"

# moderator analysis setup -----------------------------------------------------------

moderator_names

moderators = (
    [
        "n",
        "p_male",
        "p_female",
        "mean_age",
        "diagnosis",
        "sample_type",
        "current_ma",
        "sessions",
        "duration_min",
        "ramp_up_sec",
        "ramp_down_sec",
        "hemisphere",
        "site_name",
        "eeg_site",
        "measure_type",
        "measure_dimension",
        "measure_domain",
        "measure_specific",
        "measure_scale_name",
        "bias_category",
        "open_science_score",
        "methods_score",
        "reconsctruction_category",
    ],
)
continuous_expected_mod

continuous_moderators = (
    [
        "sessions",
        "duration_min",
        "p_male",
        "p_female",
        "n",
        "mean_age",
        "current_ma",
        "ramp_up_sec",
        "ramp_down_sec",
        "open_science_score",
        "methods_score",
    ],
)
dichotomous_mod

categorical_moderators = (
    [
        "diagnosis",
        "sample_type",
        "eeg_site",
        "site_name",
        "hemisphere",
        "measure_type",
        "measure_specific",
        "measure_domain",
        "measure_dimension",
        "measure_scale_name",
        "bias_category",
        "reconsctruction_category",
    ],
)

# Plotting Setup----------------------------------------------------------------------

## setting alpha maps for each
line_alpha_map = {
    "base:sham": 0.45,
    "base:active": 0.55,
    "base:contrast": 1.0,
}

fill_alpha_map = {
    "base:sham": 0.15,
    "base:active": 0.20,
    "base:contrast": 0.75,
}

labels = {
    "base:sham": "Sham",
    "base:active": "Active",
    "base:contrast": "Active > Sham",
}

colors = {
    "base:sham": "teal",
    "base:active": "orangered",
    "base:contrast": "black",
}

# DataFrame Setup-------------------------------------------
grp_map = {
    "active": 1,
    "sham": 0,
}

time_map = {
    "post": 1,
    "pre": 0,
    "diff": 3,
}

meta_base_df = pd.read_csv(
    "/home/dwinters/ascend-lab/data/tms-meta/20260926_neurostim-meta_dataframe-final.csv"
)

meta_mod_vals = pd.read_csv(
    "/home/dwinters/ascend-lab/data/tms-meta/20260901_bias_os_methods_scores.csv"
)

meta_df = pd.merge(left=meta_base_df, right=meta_mod_vals, on="ID")

meta_df["group"] = meta_df.group.map(grp_map)

meta_df["time"] = meta_df.time.map(time_map)

# - dataframe is prepped

# DataFrame Descriptives----------------------------------------
# Description of the full dataframe and saving_________
meta_report = hm.describe_meta_df(
    meta_df,
    save_dir=os.path.join(result_table_base, "base_dataframe_descriptive"),
    save_name="meta_df",
)


#### ----- Analyses ----- ####
# Cognition-----------------------------------------------------------------------------
print("\n-----------Cognition_Analysis-----------")
# Dataframe_________________________________________
cog_df = meta_df.loc[
    np.where(
        (meta_df.time == 0)
        & (meta_df.measure_category == "cognition")
        & (meta_df.group <= 1)
        & (meta_df.outcome == 1)
        | (meta_df.time == 1)
        & (meta_df.measure_category == "cognition")
        & (meta_df.group <= 1)
        & (meta_df.outcome == 1)
    )
].reset_index(drop=True)

cog_df_pp, cog_descriptives = hm.df_pre_post(cog_df)
cog_descriptives["n_post_only"] == 0  # True
##- no studies removed so all are pre - post studies

# Bayes Estimation__________________________________
## pre - post (only have pre-post)
cog_base, cog_mod, cog_info = hm.bayes_meta_hre(
    cog_df_pp,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction=">",
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_cognition"),
)

# Description of the final dataframe and saving
cog_report = hm.describe_meta_df(
    cog_info["final_df"],
    save_dir=os.path.join(result_table_base, "results_cognition"),
)


# Extracting valid moderators and saving
cog_valid_mod, cog_mod_summaries, cog_mod_levels = hm.mod_eval(
    mod_dict_primary=cog_mod,
    mod_dict_sensitivity=cog_mod,
    save_dir=os.path.join(result_table_base, "results_cognition"),
)

# Identifying Most Promising Moderator
cog_mod_pairwise = {}
cog_mod_best = {}
cog_best_levels = {}
for ii in cog_mod_levels.keys():
    pw, bst = hm.compare_moderator_levels(
        moderator_traces=cog_info["trace_mod"],
        moderator=ii,
        improvement_direction=">",
        save_dir=os.path.join(result_table_base, "results_cognition"),
        robust_levels=cog_mod_levels,
    )
    cog_mod_pairwise[ii] = pw
    cog_mod_best[ii] = bst
    # best levels
    cog_best_levels[ii] = hm.get_best_level(bst)

#

# Saving_____________________________________________
# Saving base
cog_base.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_cognition"),
        "table_base-cognition.tsv",
    ),
    sep="\t",
)

# Saving model info
with open(
    os.path.join(result_table_base, "results_cognition", "cog_info.pkl"),
    "wb",
) as file:
    pickle.dump(cog_info, file)


# Saving base figure and selected moderators figures
cog_fig_info = hm.plot_bayes_meta_results(
    cog_info,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    moderators=[
        "measure_domain",
        "measure_specific",
        "measure_scale_name",
    ],
    moderator_effects=[
        "contrast",
    ],
    moderator_levels={
        "measure_domain": ["hot_processing", "processing", "executive_function"],
        "measure_specific": ["hot_speed", "speed", "attention", "inhibition"],
        "measure_scale_name": [
            "rt_neutral",
            "rt_angry",
            "SSRT",
            "rt",
            "omission_error",
            "comission_error",
        ],
    },
    colors=colors
    | {
        "measure_scale_name:comission_error:contrast": "darkred",
        "measure_scale_name:omission_error:contrast": "darkred",
        "measure_scale_name:SSRT:contrast": "navy",
        "measure_scale_name:rt:contrast": "navy",
        "measure_scale_name:rt_neutral:contrast": "dimgray",
        "measure_scale_name:rt_angry:contrast": "dimgray",
        "measure_specific:attention:contrast": "darkred",
        "measure_specific:inhibition:contrast": "darkred",
        "measure_specific:speed:contrast": "navy",
        "measure_specific:hot_speed:contrast": "dimgray",
        "measure_domain:hot_processing:contrast": "dimgray",
        "measure_domain:processing:contrast": "navy",
        "measure_domain:executive_function:contrast": "darkred",
    },
    best_levels=cog_best_levels,
    title="Cognition",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
    save_dir=os.path.join(figures_base, "results_cognition"),
    zero_line_color="black",
)

plt.close()

# saving all moderation figures
with open(
    os.path.join(figures_base, "results_cognition", "cog_fig_info.pkl"),
    "wb",
) as file:
    pickle.dump(cog_fig_info, file)

#

# Externalizing----------------------------------------------------------------------
print("\n-----------Externalizing_Analysis-----------")
# DataFrame___________________________________________
ext_df = meta_df.loc[
    np.where(
        (meta_df.time == 0)
        & (meta_df.measure_category == "externalizing")
        & (meta_df.group <= 1)
        | (meta_df.time == 1)
        & (meta_df.measure_category == "externalizing")
        & (meta_df.group <= 1)
    )
].reset_index(drop=True)

ext_df_prpt, ext_descriptives = hm.df_pre_post(ext_df)
ext_descriptives["n_prepost_only"] == 0  # False
ext_descriptives["n_post_only"] == 0  # False
## must run post and pre-post


# Bayes Estimation ____________________________________
## pre-post only
ext_base_prpt, ext_mod_prpt, ext_info_prpt = hm.bayes_meta_hre(
    ext_df_prpt,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction="<",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_externalizing"),
    moderation=True,
)

ext_base, ext_mod, ext_info = hm.bayes_meta_hre(
    ext_df,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction="<",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_externalizing"),
    moderation=True,
)


# Description of the final dataframe and saving
ext_report = hm.describe_meta_df(
    ext_info["final_df"],
    save_dir=os.path.join(result_table_base, "results_externalizing"),
)

# Extracting valid moderators and saving
ext_valid_mod, ext_mod_summaries, ext_mod_levels = hm.mod_eval(
    mod_dict_primary=ext_mod,
    mod_dict_sensitivity=ext_mod_prpt,
    save_dir=os.path.join(result_table_base, "results_externalizing"),
)

# Identifying Most Promising Moderator
ext_mod_pairwise = {}
ext_mod_best = {}
ext_best_levels = {}
for ii in ext_mod_levels.keys():
    pw, bst = hm.compare_moderator_levels(
        moderator_traces=ext_info["trace_mod"],
        moderator=ii,
        improvement_direction="<",
        save_dir=os.path.join(result_table_base, "results_externalizing"),
        robust_levels=ext_mod_levels,
    )
    ext_mod_pairwise[ii] = pw
    ext_mod_best[ii] = bst
    # best levels
    ext_best_levels[ii] = hm.get_best_level(bst)


#

# Saving_________________________________________________________
# Saving base pre-post
ext_base.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_externalizing"),
        "table_base-externalizing.tsv",
    ),
    sep="\t",
)

ext_base_prpt.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_externalizing"),
        "table_base-externalizing_sensitivity.tsv",
    ),
    sep="\t",
)

# Saving model info
with open(
    os.path.join(result_table_base, "results_externalizing", "ext_info.pkl"),
    "wb",
) as file:
    pickle.dump(ext_info, file)

## sensitivity
with open(
    os.path.join(
        result_table_base, "results_externalizing", "ext_info_sensitivity.pkl"
    ),
    "wb",
) as file:
    pickle.dump(ext_info_prpt, file)


# Saving base figure and selected moderators figures___________
## Primary and modreations
ext_fig_info = hm.plot_bayes_meta_results(
    ext_info,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    moderators=[
        "sample_type",
        "diagnosis",
        "sessions",
        "site_name",
        "eeg_site",
        "measure_type",
        "measure_dimension",
        "measure_domain",
        "measure_specific",
        # "measure_scale_name", #due to redundancy - will move this to supplemental.
        "bias_category",
    ],
    moderator_effects=[
        "contrast",
    ],
    moderator_levels={
        "sample_type": ["clinical_forensic"],
        "diagnosis": ["subset_reported", "none_reported", "BPD"],
        "site_name": ["vmpfc", "dlpfc"],
        "eeg_site": ["Fpz", "F3_F4", "F4"],
        "measure_type": ["self_report"],
        "measure_dimension": ["state", "trait"],
        "measure_domain": ["impulsivity", "aggression"],
        "measure_specific": [
            "global_impulsivity",
            "rashness",
            "disinhibition",
            "distractability",
            "inattention",
            "hostility",
            "physical",
            "temper",
            "verbal",
        ],
        "measure_scale_name": [
            "punishment_selected",
            "premeditated_aggression",
            "aggressive_responses",
            "total_impulsivity",
            "reactive_aggression",
            "nonplanning",
            "motor",
            "total_aggression",
            "proactive_aggression",
            "NET_index",
            "cognitive_instability",
            "impulse_aggression",
            "hostility",
            "verbal_aggression",
            "aggressive_emotion",
            "physical_aggression",
        ],
        "bias_category": ["Some concerns", "High risk"],
    },
    colors=colors
    | {
        "sample_type:clinical_forensic:contrast": "darkred",
        "diagnosis:none_reported:contrast": "darkred",
        "diagnosis:BPD:contrast": "darkred",
        "diagnosis:subset_reported:contrast": "navy",
        "sessions:contrast": "darkred",
        "site_name:dlpfc:contrast": "darkred",
        "site_name:vmpfc:contrast": "navy",
        "measure_domain:aggression:contrast": "darkred",
        "measure_domain:impulsivity:contrast": "dimgray",
        "methods_score:contrast": "darkred",
        "eeg_site:F3_F4:contrast": "navy",
        "eeg_site:F4:contrast": "darkred",
        "eeg_site:Fpz:contrast": "navy",
        "hemisphere:right:contrast": "darkred",
        "hemisphere:bilateral:contrast": "darkred",
        "hemisphere:midline:contrast": "darkred",
        "measure_type:self_report:contrast": "darkred",
        "measure_specific:verbal:contrast": "darkred",
        "measure_specific:global_impulsivity:contrast": "navy",
        "measure_specific:physical:contrast": "darkred",
        "measure_specific:hostility:contrast": "darkred",
        "measure_specific:reactive:contrast": "darkred",
        "measure_specific:global_aggression:contrast": "darkred",
        "measure_specific:temper:contrast": "darkred",
        "measure_specific:rashness:contrast": "navy",
        "measure_specific:disinhibition:contrast": "navy",
        "measure_specific:inattention:contrast": "darkred",
        "measure_specific:distractability:contrast": "navy",
        "measure_dimension:state:contrast": "darkred",
        "measure_dimension:trait:contrast": "darkred",
        "bias_category:Some concerns:contrast": "darkred",
        "bias_category:High risk:contrast": "darkred",
    },
    best_levels=ext_best_levels,
    title="Externalizing",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
    save_dir=os.path.join(figures_base, "results_externalizing"),
)

plt.close()

## Sensitivity no moderation
ext_fig_info_prpt = hm.plot_bayes_meta_results(
    ext_info_prpt,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    colors=colors,
    title="Externalizing Sensitivity Analysis",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
)

plt.close()

ext_fig_info_prpt["category_combined"]["base"].savefig(
    os.path.join(
        figures_base,
        "results_externalizing",
        "effect_plot_bayes_meta_combined_base_sensitivity.tiff",
    ),
    dpi=300,
)


# saving all moderation figures____________________________
# Primary
with open(
    os.path.join(figures_base, "results_externalizing", "ext_fig_info.pkl"),
    "wb",
) as file:
    pickle.dump(ext_fig_info, file)

# Sensitivity
with open(
    os.path.join(figures_base, "results_externalizing", "ext_fig_info_sensitivity.pkl"),
    "wb",
) as file:
    pickle.dump(ext_fig_info_prpt, file)

#

# Internalizing-----------------------------------------------------------------------
print("\n-----------Internalizing_Analysis-----------")
# DataFrame___________________________________________
int_df = meta_df.loc[
    np.where(
        (meta_df.time == 0)
        & (meta_df.measure_category == "internalizing")
        & (meta_df.group <= 1)
        | (meta_df.time == 1)
        & (meta_df.measure_category == "internalizing")
        & (meta_df.group <= 1)
    )
].reset_index(drop=True)


int_df_prpt, int_descriptives = hm.df_pre_post(int_df)
int_descriptives["n_prepost_only"] == 0  # False
int_descriptives["n_post_only"] == 0  # True
## no post only studies so we have to run only pre-post studies


# Bayes Estimation ____________________________________
## pre-post only
int_base, int_mod, int_info = hm.bayes_meta_hre(
    int_df,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction="<",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_internalizing"),
    moderation=True,
)


# Description of the final dataframe and saving
int_report = hm.describe_meta_df(
    int_info["final_df"],
    save_dir=os.path.join(result_table_base, "results_internalizing"),
)


# Extracting valid moderators and saving
int_valid_mod, int_mod_summaries, int_mod_levels = hm.mod_eval(
    mod_dict_primary=int_mod,
    mod_dict_sensitivity=int_mod,
    save_dir=os.path.join(result_table_base, "results_internalizing"),
)

# Identifying Most Promising Moderator
int_mod_pairwise = {}
int_mod_best = {}
int_best_levels = {}
for ii in int_mod_levels.keys():
    pw, bst = hm.compare_moderator_levels(
        moderator_traces=int_info["trace_mod"],
        moderator=ii,
        improvement_direction="<",
        save_dir=os.path.join(result_table_base, "results_internalizing"),
        robust_levels=int_mod_levels,
    )
    int_mod_pairwise[ii] = pw
    int_mod_best[ii] = bst
    # best levels
    int_best_levels[ii] = hm.get_best_level(bst)


# Saving_____________________________________________
## primary
int_base.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_internalizing"),
        "table_base-internalizing.tsv",
    ),
    sep="\t",
)

# Saving model info
## primary
with open(
    os.path.join(result_table_base, "results_internalizing", "int_info.pkl"),
    "wb",
) as file:
    pickle.dump(int_info, file)


# Saving base figure and selected moderators figures________
# Primary with moderators
int_fig_info = hm.plot_bayes_meta_results(
    int_info,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    moderators=None,
    moderator_effects=None,
    moderator_levels=None,
    colors=colors,
    title="Internalizing",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
    save_dir=os.path.join(figures_base, "results_internalizing"),
)

plt.close()


# saving all moderation figures___________________________
# Primary
with open(
    os.path.join(figures_base, "results_internalizing", "int_fig_info.pkl"),
    "wb",
) as file:
    pickle.dump(int_fig_info, file)


# Personality-----------------------------------------------------------------------
print("\n-----------Personality_Analysis-----------")
# DataFrame___________________________________________
pers_df = meta_df.loc[
    np.where(
        (meta_df.time == 0)
        & (meta_df.measure_category == "personality")
        & (meta_df.group <= 1)
        | (meta_df.time == 1)
        & (meta_df.measure_category == "personality")
        & (meta_df.group <= 1)
    )
].reset_index(drop=True)


pers_df_prpt, pers_descriptives = hm.df_pre_post(pers_df)
pers_descriptives["n_prepost_only"] == 0  # False
pers_descriptives["n_post_only"] == 0  # False
## must run both post only and prepost


# Bayes Estimation ____________________________________
## pre-post only
pers_base_prpt, pers_mod_prpt, pers_info_prpt = hm.bayes_meta_hre(
    pers_df_prpt,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction=">",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_personality"),
    moderation=True,
)

pers_base, pers_mod, pers_info = hm.bayes_meta_hre(
    pers_df,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction=">",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_personality"),
    moderation=True,
)


# Description of the final dataframe and saving
pers_report = hm.describe_meta_df(
    pers_info["final_df"],
    save_dir=os.path.join(result_table_base, "results_personality"),
)


# Extracting valid moderators and saving
pers_valid_mod, pers_mod_summariespers_mod_levels, pers_mod_levels = hm.mod_eval(
    mod_dict_primary=pers_mod,
    mod_dict_sensitivity=pers_mod_prpt,
    save_dir=os.path.join(result_table_base, "results_personality"),
)

# Identifying Most Promising Moderator
pers_mod_pairwise = {}
pers_mod_best = {}
pers_best_levels = {}
for ii in pers_mod_levels.keys():
    pw, bst = hm.compare_moderator_levels(
        moderator_traces=pers_info["trace_mod"],
        moderator=ii,
        improvement_direction=">",
        save_dir=os.path.join(result_table_base, "results_personality"),
        robust_levels=pers_mod_levels,
    )
    pers_mod_pairwise[ii] = pw
    pers_mod_best[ii] = bst
    # best levels
    pers_best_levels[ii] = hm.get_best_level(bst)

#


# Saving_____________________________________________
# Saving base
pers_base.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_personality"),
        "table_base-personality.tsv",
    ),
    sep="\t",
)

pers_base_prpt.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_personality"),
        "table_base-personality_sensitivity.tsv",
    ),
    sep="\t",
)


# Saving model info
with open(
    os.path.join(result_table_base, "results_personality", "pers_info.pkl"),
    "wb",
) as file:
    pickle.dump(pers_info, file)


with open(
    os.path.join(result_table_base, "results_personality", "pers_info_sensitivity.pkl"),
    "wb",
) as file:
    pickle.dump(pers_info_prpt, file)


# Saving base figure and selected moderators figures______________
# Primary no moderators
pers_fig_info = hm.plot_bayes_meta_results(
    pers_info,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    moderators=None,
    moderator_effects=None,
    moderator_levels=None,
    colors=colors,
    title="Personality",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
    save_dir=os.path.join(figures_base, "results_personality"),
)

plt.close()

# Sensitivity no moderators
pers_fig_info_prpt = hm.plot_bayes_meta_results(
    pers_info_prpt,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    colors=colors,
    base_effects=["sham", "active", "contrast"],
    title="Personality Sensitivity Analysis",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
)

plt.close()

pers_fig_info_prpt["category_combined"]["base"].savefig(
    os.path.join(
        figures_base,
        "results_personality",
        "effect_plot_bayes_meta_combined_base_sensitivity.tiff",
    ),
    dpi=300,
)


# saving all moderation figures __________________________________
# Primary
with open(
    os.path.join(figures_base, "results_personality", "pers_fig_info.pkl"),
    "wb",
) as file:
    pickle.dump(pers_fig_info, file)

# Sensitivity
with open(
    os.path.join(figures_base, "results_personality", "pers_fig_info_sensitivity.pkl"),
    "wb",
) as file:
    pickle.dump(pers_fig_info_prpt, file)


# Regulation-----------------------------------------------------------------------
print("\n-----------Regulation_Analysis-----------")
# DataFrame___________________________________________
reg_df = meta_df.loc[
    np.where(
        (meta_df.time == 0)
        & (meta_df.measure_category == "regulation")
        & (meta_df.group <= 1)
        | (meta_df.time == 1)
        & (meta_df.measure_category == "regulation")
        & (meta_df.group <= 1)
    )
].reset_index(drop=True)


reg_df_prpt, reg_descriptives = hm.df_pre_post(reg_df)
reg_descriptives["n_prepost_only"] == 0  # False
reg_descriptives["n_post_only"] == 0  # False
## must run both post only and prepost


# Bayes Estimation ____________________________________
## pre-post only
reg_base_prpt, reg_mod_prpt, reg_info_prpt = hm.bayes_meta_hre(
    reg_df_prpt,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction=">",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_regulation"),
    moderation=True,
)

reg_base, reg_mod, reg_info = hm.bayes_meta_hre(
    reg_df,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction=">",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_regulation"),
    moderation=True,
)


# Description of the final dataframe and saving
reg_report = hm.describe_meta_df(
    reg_info["final_df"],
    save_dir=os.path.join(result_table_base, "results_regulation"),
)


# Extracting valid moderators and saving
reg_valid_mod, reg_mod_summaries, reg_mod_levels = hm.mod_eval(
    mod_dict_primary=reg_mod,
    mod_dict_sensitivity=reg_mod_prpt,
    save_dir=os.path.join(result_table_base, "results_regulation"),
)

# Identifying Most Promising Moderator
reg_mod_pairwise = {}
reg_mod_best = {}
reg_best_levels = {}
for ii in reg_mod_levels.keys():
    pw, bst = hm.compare_moderator_levels(
        moderator_traces=reg_info["trace_mod"],
        moderator=ii,
        improvement_direction=">",
        save_dir=os.path.join(result_table_base, "results_regulation"),
        robust_levels=reg_mod_levels,
    )
    reg_mod_pairwise[ii] = pw
    reg_mod_best[ii] = bst
    # best levels
    reg_best_levels[ii] = hm.get_best_level(bst)

#

# Saving_____________________________________________
# Saving base
reg_base.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_regulation"),
        "table_base-regulation.tsv",
    ),
    sep="\t",
)

reg_base_prpt.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_regulation"),
        "table_base-regulation_sensitivity.tsv",
    ),
    sep="\t",
)


# Saving model info
with open(
    os.path.join(result_table_base, "results_regulation", "reg_info.pkl"),
    "wb",
) as file:
    pickle.dump(reg_info, file)

with open(
    os.path.join(result_table_base, "results_regulation", "reg_info_sensitivity.pkl"),
    "wb",
) as file:
    pickle.dump(reg_info_prpt, file)


# Saving base figure and selected moderators figures
# Primary no moderators
reg_fig_info = hm.plot_bayes_meta_results(
    reg_info,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    moderators=None,
    moderator_effects=None,
    moderator_levels=None,
    colors=colors,
    title="Regulation",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
    save_dir=os.path.join(figures_base, "results_regulation"),
)

plt.close()

# Sensitivity no moderators
reg_fig_info_prpt = hm.plot_bayes_meta_results(
    reg_info_prpt,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    moderators=None,
    moderator_effects=None,
    moderator_levels=None,
    colors=colors,
    title="Regulation Sensitivity Analysis",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
)

plt.close()

reg_fig_info_prpt["category_combined"]["base"].savefig(
    os.path.join(
        figures_base,
        "results_regulation",
        "effect_plot_bayes_meta_combined_base_sensitivity.tiff",
    ),
    dpi=300,
)


# saving all moderation figures________________________
# Primary
with open(
    os.path.join(figures_base, "results_regulation", "reg_fig_info.pkl"),
    "wb",
) as file:
    pickle.dump(reg_fig_info, file)

# Sensitivity
with open(
    os.path.join(figures_base, "results_regulation", "reg_fig_info_sensitivity.pkl"),
    "wb",
) as file:
    pickle.dump(reg_fig_info_prpt, file)


# Emotion-----------------------------------------------------------------------
print("\n-----------Emotion_Analysis-----------")
# DataFrame___________________________________________
emo_df = meta_df.loc[
    np.where(
        (meta_df.time == 0)
        & (meta_df.measure_category == "emotion")
        & (meta_df.group <= 1)
        | (meta_df.time == 1)
        & (meta_df.measure_category == "emotion")
        & (meta_df.group <= 1)
    )
].reset_index(drop=True)


emo_df_prpt, emo_descriptives = hm.df_pre_post(emo_df)
emo_descriptives["n_prepost_only"] == 0  # False
emo_descriptives["n_post_only"] == 0  # False
## run both prepost and post only

# Bayes Estimation ____________________________________
emo_base_prpt, emo_mod_prpt, emo_info_prpt = hm.bayes_meta_hre(
    emo_df_prpt,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction=">",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_emotion"),
    moderation=True,
)


emo_base, emo_mod, emo_info = hm.bayes_meta_hre(
    emo_df,
    moderator_names=moderators,
    continuous_expected_mod=continuous_moderators,
    dichotomous_mod=categorical_moderators,
    improvement_direction=">",
    tacc=0.99,
    baseline_cent=True,
    measure_col="measure_name",
    direction_col="category_direction",
    # qc_plt_save_dir=os.path.join(figures_base, "qc_figures", "category_emotion"),
    moderation=True,
)


# Description of the final dataframe and saving
emo_report = hm.describe_meta_df(
    emo_info["final_df"],
    save_dir=os.path.join(result_table_base, "results_emotion"),
)


# Extracting valid moderators and saving
emo_valid_mod, emo_mod_summaries, emo_mod_levels = hm.mod_eval(
    mod_dict_primary=emo_mod,
    mod_dict_sensitivity=emo_mod_prpt,
    save_dir=os.path.join(result_table_base, "results_emotion"),
)

# Identifying Most Promising Moderator
emo_mod_pairwise = {}
emo_mod_best = {}
emo_best_levels = {}
for ii in emo_mod_levels.keys():
    pw, bst = hm.compare_moderator_levels(
        moderator_traces=emo_info["trace_mod"],
        moderator=ii,
        improvement_direction=">",
        save_dir=os.path.join(result_table_base, "results_emotion"),
        robust_levels=emo_mod_levels,
    )
    emo_mod_pairwise[ii] = pw
    emo_mod_best[ii] = bst
    # best levels
    emo_best_levels[ii] = hm.get_best_level(bst)

#

# Saving_____________________________________________
# Saving base
emo_base.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_emotion"),
        "table_base-emotion.tsv",
    ),
    sep="\t",
)

emo_base_prpt.to_csv(
    os.path.join(
        os.path.join(result_table_base, "results_emotion"),
        "table_base-emotion_sensitivity.tsv",
    ),
    sep="\t",
)


# Saving model info
with open(
    os.path.join(result_table_base, "results_emotion", "emo_info.pkl"),
    "wb",
) as file:
    pickle.dump(emo_info, file)


with open(
    os.path.join(result_table_base, "results_emotion", "emo_info_sensitivity.pkl"),
    "wb",
) as file:
    pickle.dump(emo_info_prpt, file)


# Saving base figure and selected moderators figures___________
# Primary
emo_fig_info = hm.plot_bayes_meta_results(
    emo_info,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    moderators=[
        "p_female",
        "hemisphere",
        "site_name",
        "eeg_site",
    ],
    moderator_effects=[
        "contrast",
    ],
    moderator_levels={
        "hemisphere": ["bilateral", "right", "left", "midline"],
        "site_name": ["cerebellum", "dlpfc", "ifg", "vmpfc"],
        "eeg_site": ["P10", "F3_F4", "F8", "F4", "F3", "Fpz"],
    },
    colors=colors
    | {
        "p_female:contrast": "darkred",
        "eeg_site:F3_F4:contrast": "dimgray",
        "eeg_site:F3:contrast": "navy",
        "eeg_site:F4:contrast": "navy",
        "eeg_site:P10:contrast": "dimgray",
        "eeg_site:F8:contrast": "dimgray",
        "eeg_site:Fpz:contrast": "darkred",
        "site_name:cerebellum:contrast": "dimgray",
        "site_name:dlpfc:contrast": "dimgray",
        "site_name:vmpfc:contrast": "darkred",
        "site_name:ifg:contrast": "navy",
        "hemisphere:right:contrast": "dimgray",
        "hemisphere:bilateral:contrast": "dimgray",
        "hemisphere:left:contrast": "navy",
        "hemisphere:midline:contrast": "darkred",
    },
    best_levels=emo_best_levels,
    title="emotion",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
    save_dir=os.path.join(figures_base, "results_emotion"),
)


plt.close()

# Sensitivity
emo_fig_info_prpt = hm.plot_bayes_meta_results(
    emo_info_prpt,
    line_alpha_map=line_alpha_map,
    fill_alpha_map=fill_alpha_map,
    labels=labels,
    base_effects=["sham", "active", "contrast"],
    colors=colors,
    title="emotion Sensitivity Analysis",
    xlabel="Posterior Effect",
    show=False,
    plot_mode="category_combined",
)


plt.close()

emo_fig_info_prpt["category_combined"]["base"].savefig(
    os.path.join(
        figures_base,
        "results_emotion",
        "effect_plot_bayes_meta_combined_base_sensitivity.tiff",
    ),
    dpi=300,
)


# saving all moderation figures___________________________
# Primary
with open(
    os.path.join(figures_base, "results_emotion", "emo_fig_info.pkl"),
    "wb",
) as file:
    pickle.dump(emo_fig_info, file)

# sensitivity
with open(
    os.path.join(figures_base, "results_emotion", "emo_fig_info_sensitivity.pkl"),
    "wb",
) as file:
    pickle.dump(emo_fig_info_prpt, file)

#
