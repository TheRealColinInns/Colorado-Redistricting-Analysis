import pandas as pd
import geopandas as gpd
import maup
import time

import matplotlib.pyplot as plt
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from functools import partial

maup.progress.enabled = True


def create_updaters():
    # 'PRES20D', 'PRES20R',
    # 'SEN20D', 'SEN20R',
    # 'PRES16D', 'PRES16R',
    # 'SEN16D', 'SEN16R',
    # 'GOV18D', 'GOV18R',
    # 'ATG18D', 'ATG18R',
    # 'SOS18D', 'SOS18R',
    # 'TRE18D', 'TRE18R',

    elections = [
        Election("PRES20", {"Democratic": "PRES20D", "Republican": "PRES20R"}),
        Election("SEN20", {"Democratic": "SEN20D", "Republican": "SEN20R"}),
        Election("PRES16", {"Democratic": "PRES16D", "Republican": "PRES16R"}),
        Election("SEN16", {"Democratic": "SEN16D", "Republican": "SEN16R"}),
    ]

    # 'TOTPOP20', 'VAP20', 'HISP', 'NH_WHITE', 'NH_BLACK', 'NH_AMIN', 'NH_ASIAN', 'NH_NHPI', 'NH_OTHER', 'NH_2MORE'
    chain_updaters = {"population": Tally("TOTPOP20", alias="population"),
                      "white population": Tally("NH_WHITE", alias="population"),
                      "black population": Tally("NH_BLACK", alias="population"),
                      "hispanic population": Tally("HISP", alias="population"),
                      "native population": Tally("NH_AMIN", alias="population"),
                      "asian population": Tally("NH_ASIAN", alias="population"),
                      "cut edges": cut_edges,
                      }

    election_updaters = {election.name: election for election in elections}
    chain_updaters.update(election_updaters)

    return chain_updaters


def create_chain(initial_partition, total_steps_in_run):
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

    proposal = partial(recom,
                       pop_col="TOTPOP20",
                       pop_target=ideal_population,
                       epsilon=0.02,
                       node_repeats=2
                       )

    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut edges"]),
        2 * len(initial_partition["cut edges"])
    )

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.01)

    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            pop_constraint,
            compactness_bound
        ],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=total_steps_in_run
    )

    return chain


def update_election_ensemble(part, part_key, percent, ensemble):
    win = 0
    for perc in part[part_key].percents(percent):
        if perc >= 0.5:
            win += 1

    ensemble.append(win)


def update_population_ensemble(part, part_key, num_dists, ensemble):
    num_majority = 0
    for district in range(1, num_dists + 1):
        percent = part[part_key][district] / part["population"][district]
        if percent >= 0.5:
            num_majority += 1

    ensemble.append(num_majority)


def walk(chain, num_dists):
    cut_edge_ensemble = []

    pres_16_df = []
    pres_20_df = []

    dem_pres_16_ensemble = []
    dem_pres_20_ensemble = []

    white_ensemble = []
    black_ensemble = []
    hispanic_ensemble = []
    native_ensemble = []
    asian_ensemble = []

    for part in chain.with_progress_bar():
        # Add cut edges
        cut_edge_ensemble.append(len(part["cut edges"]))

        # Add majority population by ethnic groups
        update_population_ensemble(part, "white population", num_dists, white_ensemble)
        update_population_ensemble(part, "black population", num_dists, black_ensemble)
        update_population_ensemble(part, "hispanic population", num_dists, hispanic_ensemble)
        update_population_ensemble(part, "native population", num_dists, native_ensemble)
        update_population_ensemble(part, "asian population", num_dists, asian_ensemble)

        # Add democratic majority
        update_election_ensemble(part, "PRES16", "Democratic", dem_pres_16_ensemble)
        update_election_ensemble(part, "PRES20", "Democratic", dem_pres_20_ensemble)

        pres_16_df.append(sorted(part["PRES16"].percents("Democratic")))
        pres_20_df.append(sorted(part["PRES20"].percents("Democratic")))

    ensembles = [cut_edge_ensemble,
                 pd.DataFrame(pres_16_df), pd.DataFrame(pres_20_df),
                 dem_pres_16_ensemble, dem_pres_20_ensemble,
                 white_ensemble, black_ensemble, hispanic_ensemble, native_ensemble, asian_ensemble]

    return ensembles


def create_signature_plot(data, election_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw 50% line
    ax.axhline(0.5, color="#cccccc")

    # Draw boxplot
    data.boxplot(ax=ax, positions=range(len(data.columns)))

    # Draw initial plan's Democratic vote %s (.iloc[0] gives the first row, which corresponds to the initial plan)
    plt.plot(data.iloc[0], "ro")

    # Annotate
    ax.set_title("Comparing the 2020 Congressional plan to an ensemble")
    ax.set_ylabel(f"Democratic vote % ({election_name})")
    ax.set_xlabel("Sorted districts")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    plt.savefig(f'Signature of Gerrymandering - {election_name}.png')


def create_hist(ensemble, title):
    plt.figure()
    plt.title(title)
    plt.hist(ensemble, align='left')
    plt.savefig(f'{title}.png')


def main():
    start_time = time.time()

    co_graph = Graph.from_file('./data/cleaned/final data/CO.shp')

    initial_partition = Partition(
        co_graph,
        assignment='CD',
        updaters=create_updaters()
    )

    chain = create_chain(initial_partition, total_steps_in_run=5000)

    cut_edge_ensemble, pres_16_df, pres_20_df, dem_pres_16_ensemble, dem_pres_20_ensemble, white_ensemble, \
    black_ensemble, hispanic_ensemble, native_ensemble, asian_ensemble = walk(chain, num_dists=8)

    create_hist(cut_edge_ensemble, 'Cut Edges')

    create_hist(dem_pres_16_ensemble, 'Democratic-Won Districts (2016 Presidential Election)')
    create_hist(dem_pres_20_ensemble, 'Democratic-Won Districts (2020 Presidential Election)')

    create_hist(white_ensemble, 'White-Majority Districts')
    create_hist(black_ensemble, 'Black-Majority Districts')
    create_hist(hispanic_ensemble, 'Hispanic-Majority Districts')
    create_hist(native_ensemble, 'Native-Majority Districts')
    create_hist(asian_ensemble, 'Asian-Majority Districts')

    create_signature_plot(pres_16_df, "2016 Presidential")
    create_signature_plot(pres_20_df, "2020 Presidential")

    end_time = time.time()
    print(f"The time of execution of above program is: {(end_time - start_time) / 60} mins")


if __name__ == '__main__':
    main()
