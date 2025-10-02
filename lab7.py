import matplotlib.pyplot as plt
import networkx as nx
from instagrapi import Client
from instagrapi.exceptions import TwoFactorRequired

MAX_FOLLOWINGS_COUNT = 20  # Maximum number of followings to fetch

# Login to Instagram
instagram_client = Client()
instagram_client.delay_range = [1, 5]  # Recommended delay to avoid rate limits

USERNAME = input("Input your Instagram username: ")
PASSWORD = input("Input your Instagram password: ")

assert USERNAME, 'Username must be provided'
assert PASSWORD, 'Password must be provided'

try:
    instagram_client.login(USERNAME, PASSWORD)
    print("Logged in successfully")
except TwoFactorRequired:
    print("Two-factor authentication required. Please disable it in your Instagram settings.")
    raise

# Fetch followings of the user
my_followings = instagram_client.user_following(user_id=instagram_client.user_id,
                                                amount=MAX_FOLLOWINGS_COUNT)
my_followings_names = [user.username for user in my_followings.values()]

# Build graph
G = nx.Graph()
G.add_node(instagram_client.username, label=instagram_client.username)

# Add edges from user to their followings
for following in my_followings.values():
    G.add_node(following.username, label=following.full_name)
    G.add_edge(instagram_client.username, following.username)

# Add edges between followings if they follow each other
for person in my_followings.values():
    try:
        print(f'Processing followings of: {person.username}')
        following_followings = instagram_client.user_following(person.pk, amount=MAX_FOLLOWINGS_COUNT)
        for following in following_followings.values():
            if following.username in my_followings_names:
                G.add_node(following.username, label=following.full_name)
                G.add_edge(person.username, following.username)
    except Exception as e:
        print(f"Error fetching data for {person.username}: {e}")

#  Save the graph
nx.write_gexf(G, "InstaFriends.gexf")
print("Graph saved as InstaFriends.gexf")

#  Compute average clustering coefficient
avg_clustering = nx.average_clustering(G)
print(f"Average clustering coefficient: {avg_clustering:.4f}")

#  Visualize the graph
plt.figure(figsize=(12, 8))
nx.draw_spring(G, with_labels=True, font_weight='bold', font_size=5, node_color='skyblue', edge_color='gray')
plt.title("Instagram Followers Graph")
plt.savefig('InstaGraf.png', dpi=600)
plt.show()
