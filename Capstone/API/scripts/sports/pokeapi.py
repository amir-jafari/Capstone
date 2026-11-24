META = dict(
    name="PokéAPI",
    domain="Sports",
    docs="https://pokeapi.co/docs/v2",
    auth="None",
    signup="No registration. Fair use: cache responses locally (the data rarely changes). Also available as a GraphQL "
           "endpoint (beta) and a full data dump.",
    provides=["Every Pokémon: base stats, types, abilities, height/weight, moves, sprites",
              "Species (evolution chains, habitats, flavor text in many languages), types and damage relations, moves, items",
              "Generations and games"],
    ideas="Classic teaching dataset for clustering/classification, type-matchup graph analysis, team optimization.",
    endpoint="GET https://pokeapi.co/api/v2/{resource}/{id or name}",
    params=[
        ("pokemon/{name}", "One Pokémon (stats, types, abilities)", "pikachu"),
        ("pokemon?limit=&offset=", "List all Pokémon (count in response)", "limit=100000"),
        ("type/{name}", "Type with damage relations and every Pokémon of that type", "fire"),
        ("pokemon-species/{name} · evolution-chain/{id}", "Species info and evolutions", "—"),
        ("generation/{id}", "Pokémon introduced in a generation", "1"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show

API = "https://pokeapi.co/api/v2"


def main():
    p = get(f"{API}/pokemon/pikachu")
    count = get(f"{API}/pokemon", params={"limit": 1})["count"]
    fire = get(f"{API}/type/fire")
    show({"total_pokemon_entries": count,
          "pikachu": {"types": [t["type"]["name"] for t in p["types"]], "height_dm": p["height"], "weight_hg": p["weight"],
                      "base_stats": {s["stat"]["name"]: s["base_stat"] for s in p["stats"]}},
          "fire_type": {"pokemon": len(fire["pokemon"]),
                        "double_damage_to": [t["name"] for t in fire["damage_relations"]["double_damage_to"]],
                        "weak_to": [t["name"] for t in fire["damage_relations"]["double_damage_from"]]}})


if __name__ == "__main__":
    main()
