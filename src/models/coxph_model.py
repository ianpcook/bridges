import numpy as np
import pandas as pd
import lifelines as ll

inpath = '/Users/ian/Documents/exploratory/bridges/data/processed/'
filename = 'bridges_coxph.csv'


def get_df(inpath, filename):
    return pd.read_csv(inpath+filename)


def fit_model():
    df = get_df(inpath, filename)
    # dropping columns for reference categories
    df.drop(['Concrete', 'Urban'], axis=1, inplace=True)
    coxph = ll.CoxPHFitter()
    model = coxph.fit(df, duration_col='duration',
                     event_col='degraded_obs',
                     cluster_col='id',
                     show_progress=True)
    pprint(model.summary)
    return model


def basic_survival(df):
    T = df["duration"]
    E = df["degraded_obs"]
    kmf = ll.KaplanMeierFitter()
    model = kmf.fit(durations=T, event_observed=E)
    model.plot(figsize=(9,8))
    plt.title('Survival Function of Bridges over Time: Pooled Data Across all Bridges', fontsize=18)
    plt.savefig('/Users/ian/Documents/exploratory/bridges/reports/figures/basic_survival.png')
    plt.show()
    plt.clf()
    plt.close()


def concrete_steel_plot(df):
    ax = plt.subplot(111)
    concrete = ((df['P/S'] != 1) & (df['Steel'] != 1))
    steel = (df['Steel'] == 1)
    ps = (df['P/S'] == 1)
    concrete_model = kmf.fit(durations=T[concrete], event_observed=E[concrete], label='Concrete Bridges')
    concrete_model.plot(ax=ax, figsize=(12,10))
    steel_model = kmf.fit(durations=T[steel], event_observed=E[steel], label='Steel Bridges')
    steel_model.plot(ax=ax)
    ps_model = kmf.fit(durations=T[ps], event_observed=E[ps], label='P/S Bridges')
    ps_model.plot(ax=ax)
    plt.ylim(0, 1)
    plt.title('Survival Function of Bridges over Time: Concrete vs. Steel vs P/S Bridges', fontsize=20)
    plt.savefig('/Users/ian/Documents/exploratory/bridges/reports/figures/c_s_ps_comparison.png')
    plt.show()
    plt.clf()
    plt.close()


def hazard_plot(df):
    T = df["duration"]
    E = df["degraded_obs"]
    naf = ll.NelsonAalenFitter()
    naf.fit(T, event_observed=E)
    ax = plt.subplot(111)
    concrete = ((df['P/S'] != 1) & (df['Steel'] != 1))
    steel = (df['Steel'] == 1)
    ps = (df['P/S'] == 1)
    concrete_model = naf.fit(durations=T[concrete], event_observed=E[concrete], label='Concrete Bridges')
    concrete_model.plot_hazard(bandwidth=10, ax=ax, figsize=(12, 10))
    steel_model = naf.fit(durations=T[steel], event_observed=E[steel], label='Steel Bridges')
    steel_model.plot_hazard(bandwidth=10, ax=ax)
    ps_model = naf.fit(durations=T[ps], event_observed=E[ps], label='P/S Bridges')
    ps_model.plot_hazard(bandwidth=10, ax=ax)
    plt.title('Hazard Functions: Concrete vs Steel vs P/S', fontsize=20)
    plt.savefig('/Users/ian/Documents/exploratory/bridges/reports/figures/hazard_c_s_ps.png')
    plt.show()
    plt.clf()
    plt.close()


def cum_hazard_plot(df):
    T = df["duration"]
    E = df["degraded_obs"]
    naf = ll.NelsonAalenFitter()
    naf.fit(T, event_observed=E)
    ax = plt.subplot(111)
    concrete = ((df['P/S'] != 1) & (df['Steel'] != 1))
    steel = (df['Steel'] == 1)
    ps = (df['P/S'] == 1)
    concrete_model = naf.fit(durations=T[concrete], event_observed=E[concrete], label='Concrete Bridges')
    concrete_model.plot(ax=ax, figsize=(12, 10))
    steel_model = naf.fit(durations=T[steel], event_observed=E[steel], label='Steel Bridges')
    steel_model.plot(ax=ax)
    ps_model = naf.fit(durations=T[ps], event_observed=E[ps], label='P/S Bridges')
    ps_model.plot(ax=ax)
    plt.title('Cumulative Hazard Functions: Concrete vs Steel vs P/S', fontsize=20)
    plt.savefig('/Users/ian/Documents/exploratory/bridges/reports/figures/cum_hazard_c_s_ps.png')
    plt.show()
    plt.clf()
    plt.close()


def predictions():
    model = fit_model()
    X = df.drop(['duration', 'degraded_obs'], axis=1)
    model.predict_median(X)
    model.predict_partial_hazard(X)
    survs = model.predict_survival_function(X)
    survs.loc[:, (0, 409, 1447)].plot(figsize=(5,4))
    plt.title('Predicted Survival Functions for 3 Bridges')
    plt.savefig('/Users/ian/Documents/exploratory/bridges/reports/figures/pred_surv.png')
    plt.clf()
    plt.close()
    plt.show()
    chaz = model.predict_cumulative_hazard(X)
    chaz.loc[:, (0, 409, 1447)].plot(figsize=(5,4))
    plt.title('Predicted Cumulative Hazards for 3 Bridges')
    plt.savefig('/Users/ian/Documents/exploratory/bridges/reports/figures/pred_chaz.png')
    plt.show()
    plt.clf()
    plt.close()
