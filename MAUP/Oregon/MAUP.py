#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:32:51 2023

@author: eveomett

Lab 3: MAUP and data.  See details on Canvas page

Make sure to say where/when you got your data!
"""

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
    elections = [
        Election("G20USS", {"Democratic": "G20USSD", "Republican": "G20USSR"})
    ]

    chain_updaters = {"population": Tally("TOTPOP", alias="population"),
                   "cut edges": cut_edges
                   }

    election_updaters = {election.name: election for election in elections}
    chain_updaters.update(election_updaters)

    return chain_updaters


def create_chain(initial_partition, pop_percent, total_steps_in_run):
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

    proposal = partial(recom,
                       pop_col="TOTPOP",
                       pop_target=ideal_population,
                       epsilon=0.02,
                       node_repeats=2
                       )

    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut edges"]),
        2 * len(initial_partition["cut edges"])
    )

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, pop_percent)

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


def walk(chain):
    cut_edge_ensemble = []
    dem_ensemble = []

    for part in chain.with_progress_bar():
        # Add cut edges
        cut_edge_ensemble.append(len(part["cut edges"]))

        dem_win = 0
        for dem_perc in part["G20USS"].percents("Democratic"):
            if dem_perc >= 0.5:
                dem_win += 1
        dem_ensemble.append(dem_win)

    return cut_edge_ensemble, dem_ensemble


def create_hist(ensemble, title):
    plt.figure()
    plt.title(title)
    plt.hist(ensemble, align='left')
    plt.savefig(f'{title}.png')


def load_data(file_name):
    start_time = time.time()
    df = gpd.read_file(file_name)
    end_time = time.time()
    print(f'The time to import {file_name} is: {(end_time - start_time) / 60} mins')

    return df


def maup_repair():
    population_df = load_data('./or_pl2020_b/or_pl2020_b.shp')
    election_df = load_data('./or_vest_20/or_vest_20.shp')
    sen_district_df = load_data('./or_sldu_2021/Senate_LC_Draft_2_-_Revised_.shp')

    district_col_name = 'DISTRICT'
    blocks_to_precincts_assignment = maup.assign(population_df.geometry.to_crs(2913), election_df.geometry)

    pop_column_names = ['P0020001', 'P0020002', 'P0020005', 'P0020006', 'P0020007',
                        'P0020008', 'P0020009', 'P0020010', 'P0020011']

    for name in pop_column_names:
        election_df[name] = population_df[name].groupby(blocks_to_precincts_assignment).sum()

    print(maup.doctor(election_df))
    repaired_election_df = maup.smart_repair(election_df, snap_precision=8)
    print(maup.doctor(repaired_election_df))

    precincts_to_districts_assignment = maup.assign(repaired_election_df.geometry,
                                                    sen_district_df.geometry.to_crs(2913))

    repaired_election_df['SEN'] = precincts_to_districts_assignment

    print(set(repaired_election_df['SEN']))
    for precinct_index in range(len(repaired_election_df)):
        repaired_election_df.at[precinct_index, 'SEN'] = sen_district_df.at[
            repaired_election_df.at[precinct_index, 'SEN'], district_col_name]
    print(set(sen_district_df[district_col_name]))
    print(set(repaired_election_df['SEN']))

    rename_dict = {'P0020001': 'TOTPOP', 'P0020002': 'HISP', 'P0020005': 'NH_WHITE', 'P0020006': 'NH_BLACK',
                   'P0020007': 'NH_AMIN',
                   'P0020008': 'NH_ASIAN', 'P0020009': 'NH_NHPI', 'P0020010': 'NH_OTHER', 'P0020011': 'NH_2MORE',
                   'P0040001': 'VAP', 'P0040002': 'HVAP', 'P0040005': 'WVAP', 'P0040006': 'BVAP', 'P0040007': 'AMINVAP',
                   'P0040008': 'ASIANVAP', 'P0040009': 'NHPIVAP', 'P0040010': 'OTHERVAP', 'P0040011': '2MOREVAP',
                   'G20PREDBID': 'G20PRED', 'G20PRERTRU': 'G20PRER', 'G20USSDMER': 'G20USSD',
                   'G20USSRPER': 'G20USSR'}
    repaired_election_df.rename(columns=rename_dict, inplace=True)

    final_election_df = repaired_election_df.drop(columns=['G20PRELJOR', 'G20PREGHAW', 'G20PREPHUN', 'G20PREOWRI',
                                                           'G20USSLDYE', 'G20USSGTAH', 'G20USSOWRI', 'G20ATGDROS',
                                                           'G20ATGRCRO', 'G20ATGLHED', 'G20ATGOWRI', 'G20SOSDFAG',
                                                           'G20SOSRTHA', 'G20SOSLMAR', 'G20SOSGPAR', 'G20SOSOWRI',
                                                           'G20TREDREA', 'G20TRERGUD', 'G20TREIHEN', 'G20TRECMAR',
                                                           'G20TREOWRI'])
    print(final_election_df.columns)

    pop_vals = [final_election_df.loc[final_election_df['SEN'] == n, 'TOTPOP'].sum() for n in range(1, 31)]
    print(pop_vals)

    final_election_df.to_file("./OR/OR_1.shp")
    shp_file = gpd.read_file('./OR/OR_1.shp')
    shp_file.to_file('./OR/OR_1.json', driver='GeoJSON')


def main():
    # It takes multiple hours to do the load and repair, so refer to OR_MAUP.ipynb
    do_maup = False

    if do_maup:
        maup_repair()

    start_time = time.time()

    or_graph = Graph.from_file('./OR/OR_1.json')

    initial_partition = Partition(
        or_graph,
        assignment='SEN',
        updaters=create_updaters()
    )

    chain = create_chain(initial_partition, 0.1, total_steps_in_run=100)

    cut_edge_ensemble, dem_ensemble = walk(chain)

    create_hist(cut_edge_ensemble, 'Cut Edges')
    create_hist(dem_ensemble, 'Democratic-Won Districts')

    end_time = time.time()
    print(f"The time of execution of above program is: {(end_time - start_time) / 60} mins")


if __name__ == '__main__':
    main()
