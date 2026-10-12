META = dict(
    name="Lemmy (federated Reddit-like communities)",
    domain="Social Media",
    docs="https://join-lemmy.org/api/main",
    auth="None for public data (per instance)",
    signup="No registration to read. Lemmy is open-source and federated: query any instance (lemmy.world, lemmy.ml, "
           "...). Listings include communities from other federated servers.",
    provides=["Posts with score, upvotes/downvotes, comment counts, community, timestamps, links",
              "Comment trees, communities with subscriber counts, user profiles", "Sorting by Hot, TopDay, TopWeek, New, ..."],
    ideas="Decentralized-community studies, post-Reddit-API migration, comparing discussion with Reddit on the same topics.",
    endpoint="GET https://{instance}/api/v3/{post/list | comment/list | community/list | search}",
    params=[
        ("instance", "Any Lemmy server", "lemmy.world"),
        ("sort", "Hot | Active | New | TopDay | TopWeek | TopMonth | TopAll | MostComments", "TopDay"),
        ("community_name", "Restrict to a community (name@instance for remote ones)", "technology"),
        ("type_", "All | Local | Subscribed", "All"),
        ("limit / page", "Max 50 per page", "10"),
        ("comment/list?post_id=", "Comments for a post", "—"),
        ("search?q=&type_=", "Search posts, comments, communities", "q=climate&type_=Posts"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    posts = get("https://lemmy.world/api/v3/post/list", params={"sort": "TopWeek", "community_name": "technology",
                                                               "limit": 10})["posts"]
    comms = get("https://lemmy.world/api/v3/community/list", params={"sort": "TopAll", "type_": "Local", "limit": 5})["communities"]
    show({"top_technology_posts_this_week": [{"title": p["post"]["name"], "community": p["community"]["name"],
                               "score": p["counts"]["score"], "comments": p["counts"]["comments"]} for p in posts[:5]],
          "largest_local_communities": [{"name": c["community"]["name"], "subscribers": c["counts"]["subscribers"]}
                                        for c in comms]})


if __name__ == "__main__":
    main()
