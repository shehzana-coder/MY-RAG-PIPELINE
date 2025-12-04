from query import perform_query
import time
from datetime import datetime

# User query
user_query = "tell me about the rector of the namal?"

# Record start time
start_time = time.time()

# Perform the query
response = perform_query(user_query)

# Record end time
end_time = time.time()
elapsed_time = end_time - start_time  # in seconds

# Print results
print(f"Query timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"User query: {user_query}\n")
print(f"Response:\n{response}\n")
print(f"Time taken: {elapsed_time:.2f} seconds")

