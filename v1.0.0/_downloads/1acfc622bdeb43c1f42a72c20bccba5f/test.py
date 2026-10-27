"""
# Within-Trial Spatial Variability (WTSpat)
"""

wt_spat_left = nearby.metrics.within_trial_spatial(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
)
wt_spat_right = nearby.metrics.within_trial_spatial(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
)
wt_spat = pd.concat([wt_spat_left, wt_spat_right])
results["WTSpat"] = wt_spat["within_trial_spatial"].mean()
print(pd.DataFrame(results, index=[0]))

"""
# Within-Trial Frequency Variability (WTFreq)
"""
wt_freq_left = nearby.metrics.within_trial_frequency(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
)

wt_freq_right = nearby.metrics.within_trial_frequency(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
)

wt_freq = pd.concat([wt_freq_left, wt_freq_right])
results["WTFreq"] = wt_freq["within_trial_frequency"].mean()
print(pd.DataFrame(results, index=[0]))

"""
Between Trial Variability
-------------------------
"""
bt_temp_left = nearby.metrics.between_trial_frequency(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)
bt_temp_right = nearby.metrics.between_trial_temporal(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)

bt_temp = pd.concat([bt_temp_left, bt_temp_right])
results["BTTemp"] = bt_temp["between_trial_temporal"].mean()
print(pd.DataFrame(results, index=[0]))

exit()

# %%
"""
# Print results
"""

print("# Variability Metrics")
print(f" - ME    : {me:10.6f}")
print(f" - SMEAT : {smeat:10.6f}")
print(f" - WTTV  : {wttv:10.6f}")
print(f" - ATTV  : {attv:10.6f}")
print(f" - ATSVA : {atsva:10.6f}")
print(f" - ATFVA : {atfva:10.6f}")
