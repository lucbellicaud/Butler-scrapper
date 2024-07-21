import json
from typing import Dict, FrozenSet, List, Set
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from utils import IMP, round_to_nearest_10, MatchInfo
import os


def get_first_board_number(
    competition_url: str,
    competition_magic_number: int,
    match_number: int,
    number_of_boards: int,
) -> int:
    board_url = "{}BoardAcross.asp?qboard={}.{}..{}".format(
        competition_url,
        f"001",
        f"{match_number:02d}",
        competition_magic_number,
    )
    list_of_df = pd.read_html(board_url)
    if len(list_of_df) >= 5:
        return 1
    modulo_possibilities = [10, 12, 14, 16, 18]
    for i in range(1, 4):
        for modulo in modulo_possibilities:
            board_url = "{}BoardAcross.asp?qboard={}.{}..{}".format(
                competition_url,
                f"{modulo*i + 1:03d}",
                f"{match_number:02d}",
                competition_magic_number,
            )
            list_of_df = pd.read_html(board_url)
            if len(list_of_df) >= 5:
                return modulo * i + 1
    raise Exception("No first board found")


def read_board(
    competition_url: str,
    competition_magic_number: int,
    match_number: int,
    board_number: int,
    full_link: str | None = None,
    simplified: bool = False,
):
    board_url = "{}BoardAcross.asp?qboard={}.{}..{}".format(
        competition_url,
        f"{board_number:03d}",
        f"{match_number:02d}",
        competition_magic_number,
    )
    if full_link is not None:
        board_url = full_link
    print(board_url)
    list_of_df = pd.read_html(board_url)
    if len(list_of_df) < 5:
        return None
    df = list_of_df[4].copy()
    # Define the first line as the header and delete it
    df.columns = df.iloc[0]
    df = df[1:]
    # print(df)

    df.loc[:, "NS"] = pd.to_numeric(df["NS"], errors="coerce")
    df.loc[:, "EW"] = pd.to_numeric(df["EW"], errors="coerce")

    # Create the new column
    df.loc[:, "scores"] = np.where(df["NS"].notna(), df["NS"], -df["EW"])

    # We have to discard the top 10% and the bottom 10% of the scores to get the average score
    score_copy = df["scores"].copy()
    score_copy.sort_values(inplace=True)
    score_copy = score_copy.iloc[
        int(len(score_copy) * 0.1) : int(len(score_copy) * 0.9)
    ]
    average_score = score_copy.mean(skipna=True)

    datum = round_to_nearest_10(average_score)
    # print("datum: {:.2f}".format(datum))

    df.loc[:, "scores"] = pd.to_numeric(df["scores"], errors="coerce")
    tab = []
    for score in df["scores"]:
        if np.isnan(score):
            tab.append(np.nan)
        else:
            tab.append(IMP(score, datum))
    df.loc[:, "IMPs to datum"] = tab

    # Rename column Cont. into contract
    df.loc[:, "contract"] = df.loc[:, "Cont."].str.split().str[0]

    # Convert the table number to an integer
    df.loc[:, "Table"] = df.loc[:, "Table"].astype(int)

    if simplified:
        df = df[["Table", "Home Team", "Visiting Team", "Room", "IMPs to datum"]]
    return df


def get_the_players(
    competition_url: str | None,
    match_id: int | None,
    home_team: str,
    away_team: str,
    table_number: int,
    full_link: str | None = None,
):
    assert competition_url is not None or full_link is not None
    url = "{}/Asp/BoardDetails.asp?qmatchid={}".format(competition_url, match_id)
    if full_link is not None:
        url = full_link
    # url = "http://db.eurobridge.org/repository/competitions/24Herning/microsite/Asp/BoardDetails.asp?qmatchid=119334"
    list_of_df = pd.read_html(url)
    players_df = list_of_df[5]
    players_df.columns = players_df.iloc[0]
    players_df = players_df[1:]
    # print(players_df)
    pair_ns_open = {str(players_df.iloc[0, 1]), str(players_df.iloc[2, 1])}
    pair_ns_close = {str(players_df.iloc[0, 4]), str(players_df.iloc[2, 4])}
    pair_ew_open = {str(players_df.iloc[1, 0]), str(players_df.iloc[1, 2])}
    pair_ew_close = {str(players_df.iloc[1, 4]), str(players_df.iloc[1, 6])}

    return MatchInfo(
        table_number,
        home_team,
        away_team,
        pair_ns_open,
        pair_ns_close,
        pair_ew_open,
        pair_ew_close,
    )


def update_dataframe_with_players(df: pd.DataFrame, match_info_list: List[MatchInfo]):
    for match in match_info_list:
        # Find the rows matching the table number and teams
        open_mask = (
            (df["Home Team"] == match.home_team)
            & (df["Visiting Team"] == match.away_team)
            & (df["Room"] == "Open")
        )
        closed_mask = (
            (df["Home Team"] == match.home_team)
            & (df["Visiting Team"] == match.away_team)
            & (df["Room"] == "Closed")
        )

        # Update the DataFrame with player names
        df.loc[open_mask, "Home Team"] = ", ".join(match.pair_ns_open)
        df.loc[open_mask, "Visiting Team"] = ", ".join(match.pair_ew_open)
        df.loc[closed_mask, "Home Team"] = ", ".join(match.pair_ns_close)
        df.loc[closed_mask, "Visiting Team"] = ", ".join(match.pair_ew_close)


def get_the_round_meta_data(
    competition_url: str, competition_magic_number: int, round_number: int
) -> pd.DataFrame:
    # Example : http://db.eurobridge.org/repository/competitions/24Herning/microsite/Asp/RoundTeams.asp?qtournid=2410&qroundno=1
    url = f"{competition_url}/RoundTeams.asp?qtournid={competition_magic_number}&qroundno={round_number}"
    response = requests.get(url)
    response.encoding = "ISO-8859-1"

    soup = BeautifulSoup(response.content, "html.parser")

    list_of_df = pd.read_html(response.text)
    df = list_of_df[3]
    # Define the first line as the header and delete it
    df.columns = df.iloc[0]
    df = df[1:]
    df.columns = df.iloc[0]
    df = df[1:]
    links = []
    # Transform the table number into an integer
    df["Table"] = df["Table"].astype(int)
    # Rename column 2 into Home Team
    # print(df)
    df.columns = [
        "Table",
        "Home Team",
        "Visiting Team",
        "Score Home Team",
        "Score Visiting Team",
        "VP Home Team",
        "VP Visiting Team",
    ]

    # Extract the links from the table - very beautiful soup indeed
    table_column = soup.find_all("table")[2].find_all("tr")
    for row in table_column:
        link = row.find("a")
        if (
            link
            and "BoardDetails" in link.get("href")
            and link.get("href") not in links
        ):
            links.append(link.get("href"))

    df["Link"] = links[: len(df)]

    # Display the DataFrame
    # print(df)

    return df


def extract_full_round_data(
    competition_url: str,
    competition_magic_number: int,
    round_number: int,
    use_cache: bool = True,
    tournament_name: str = "tournament",
    number_of_boards: int = 16,
):
    if use_cache:
        try:
            with open(
                "{}/data round {}.json".format(tournament_name, round_number), "r"
            ) as f:
                data = json.load(f)
                return {frozenset(eval(key)): value for key, value in data.items()}
        except FileNotFoundError:
            pass
    # Extract the meta data
    meta_data = get_the_round_meta_data(
        competition_url, competition_magic_number, round_number
    )
    # For each match, extract the board data, where the link can be found in the link column of the meta data
    boards_data = {}
    players = []
    for index, row in meta_data.iterrows():
        # print(row)
        if row["Home Team"] == "Bye" or row["Visiting Team"] == "Bye":
            continue
        players.append(
            get_the_players(
                competition_url=None,
                match_id=None,
                table_number=row["Table"],
                home_team=row["Home Team"],
                away_team=row["Visiting Team"],
                full_link=competition_url + row["Link"],
            )
        )
    first_board_number = get_first_board_number(
        competition_url, competition_magic_number, round_number, number_of_boards
    )
    for i in range(number_of_boards):
        board_data = read_board(
            competition_url,
            competition_magic_number,
            round_number,
            board_number=i + first_board_number,
            simplified=True,
        )
        if board_data is None:
            continue
        update_dataframe_with_players(board_data, players)
        boards_data[i] = board_data
        # print(board_data)

    imp_per_board_per_pair: Dict[FrozenSet[str], List[int]] = {}
    for i, board_data in boards_data.items():
        for index, row in board_data.iterrows():
            pair_ns = set(row["Home Team"].split(", "))
            pair_ew = set(row["Visiting Team"].split(", "))
            if frozenset(pair_ns) not in imp_per_board_per_pair:
                imp_per_board_per_pair[frozenset(pair_ns)] = [0] * len(boards_data)
            if frozenset(pair_ew) not in imp_per_board_per_pair:
                imp_per_board_per_pair[frozenset(pair_ew)] = [0] * len(boards_data)
            imp_per_board_per_pair[frozenset(pair_ns)][i] = row["IMPs to datum"]
            imp_per_board_per_pair[frozenset(pair_ew)][i] = -row["IMPs to datum"]

    print(imp_per_board_per_pair)
    # Reorder the dict based on the average IMPs
    imp_per_board_per_pair = dict(
        sorted(imp_per_board_per_pair.items(), key=lambda x: -np.average(x[1]))
    )

    # Save the data as a csv file
    df = pd.DataFrame(imp_per_board_per_pair).T
    # Add a line that is the average of numerical values
    df["Average"] = df.mean(axis=1)
    # Round the average to 2 decimal places
    df["Average"] = df["Average"].round(2)

    df.columns = [f"Board {i}" for i in range(1, number_of_boards + 1)] + ["Average"]
    # Transform the key from frozen set to a string, each value being separated by a dash
    df.index = [" - ".join(key) for key in df.index]

    df.to_csv("{}/data round {}.csv".format(tournament_name, round_number))

    # for pair, imps in imp_per_board_per_pair.items():
    #     print(pair, np.average(imps))

    # Store the data in a json file
    with open("{}/data round {}.json".format(tournament_name, round_number), "w") as f:
        # Convert the keys from frozen set to str to be able to store the data in a json file
        json.dump(
            {str(key): value for key, value in imp_per_board_per_pair.items()},
            f,
        )

    return imp_per_board_per_pair


def extract_all_the_data(
    competition_url: str,
    competition_magic_number: int,
    round_number_to_extract: List[int] | int,
    number_of_boards: int,
    use_cache: bool = True,
    tournament_name: str = "tournament",
) -> None:
    
    """This is the main function used to collect all the data from a competition. It will extract the data from all the rounds and store it in a folder named tournament_name. The data will be stored in csv files.

    Args:
        competition_url (str): The link of the microsite. Should end with microsite/. Example : http://db.eurobridge.org/repository/competitions/24wroclaw/microsite
        competition_magic_number (int): Each competition has a "magic number" that is used to identify it. You can find it by clicking on a round, you'll find it at the end of the URL. Example : 2430
        round_number_to_extract (List[int]) or int : The list of rounds to extract. You can pass a list to extract some precise rounds, or an integer to exract all the round <= to this round number. Example : [1, 2, 3, 4, 5] or 5
        number_of_boards (int, optional): Number of boards played by round.
        use_cache (bool, optional): If you already extracted the competition, you can use the cache to not request the data again. Defaults to True.
        tournament_name (str, optional): Name of the folder where you are going to store you data. Defaults to "tournament".

    Returns None and store all the data into csv files in the folder tournament_name.
    """

    if not os.path.exists(tournament_name):
        os.makedirs(tournament_name)
    if isinstance(round_number_to_extract, int):
        round_number_to_extract = list(range(1, round_number_to_extract + 1))
    assert isinstance(round_number_to_extract, list)
    average_data = {}
    # full_data = {}
    for i, round_number in enumerate(round_number_to_extract):
        round_data = extract_full_round_data(
            competition_url,
            competition_magic_number,
            round_number,
            use_cache=use_cache,
            tournament_name=tournament_name,
            number_of_boards=number_of_boards,
        )
        for pair, imps in round_data.items():
            if pair not in average_data:
                average_data[pair] = np.full(len(round_number_to_extract), np.nan)
                # full_data[pair] = np.full(len(round_number_to_extract), np.nan)
            average_data[pair][i] = np.average(imps)
            # full_data[pair][i] = imps

    # Reorder the dict based on the reversed average IMPs
    average_data = dict(sorted(average_data.items(), key=lambda x: -np.nanmean(x[1])))

    for pair, imps in average_data.items():
        # print the average while ignore the nan
        print(pair, np.nanmean(imps))
    # Add the average as a new column
    for pair, imps in average_data.items():
        average_data[pair] = np.concatenate([imps, [np.nanmean(imps)]])
    # Add the number of match played as a new column
    for pair, imps in average_data.items():
        average_data[pair] = np.concatenate(
            [imps[:], [np.count_nonzero(~np.isnan(imps[:-1]))]]
        )

    # Transform the key from frozen set to a string, each value being separated by a dash
    average_data = {
        "{} - {}".format(min(key), max(key)): value
        for key, value in average_data.items()
    }

    # Export all the data to a csv file

    df = pd.DataFrame(average_data).T
    df.columns = [f"Round {i}" for i in round_number_to_extract] + [
        "Average",
        "Number of matches",
    ]

    # Round all the values to 2 decimal places
    df = df.round(2)

    ##Move to the bottom the pairs that have not played one third of the matches
    # Create a mask to filter the pairs that have played less than 1/3 of the matches
    mask = df["Number of matches"] < len(round_number_to_extract) / 3
    # Create a new DataFrame with the pairs that have played less than 1/3 of the matches
    df_bottom = df[mask]
    # Create a new DataFrame with the pairs that have played at least 1/3 of the matches
    df = df[~mask]
    # Concatenate df and df bottom, with an empty line between them
    df = pd.concat([df, pd.DataFrame(index=[""] * 3), df_bottom])

    # Create the folder if it does not exist
    df.to_csv("{}/data.csv".format(tournament_name))

    return None


extract_all_the_data(
    "http://db.eurobridge.org/repository/competitions/24wroclaw/microsite/Asp/",
    2434,
    13,
    number_of_boards=16,
    # list(range(1, 4)),
    # use_cache=False,
    tournament_name="U31 Wroclaw",
)


# extract_full_round_data(
#     "http://db.eurobridge.org/repository/competitions/24Herning/microsite/Asp/",
#     2410,
#     1,
# )

# get_the_round_meta_data(
#     "http://db.eurobridge.org/repository/competitions/24Herning/microsite/Asp", 2410, 1
# )

# board_data = read_board(
#     "http://db.eurobridge.org/repository/competitions/24Herning/microsite/Asp",
#     2410,
#     1,
#     1,
#     simplified=True,
# )

# players = get_the_players(
#     "http://db.eurobridge.org/repository/competitions/24Herning/microsite/Asp",
#     119334,
#     "NORWAY",
#     "DENMARK",
# )
# print(players)
# update_dataframe_with_players(board_data, [players])
# print(board_data)
