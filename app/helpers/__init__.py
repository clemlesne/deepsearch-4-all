from os import getenv

from dotenv import find_dotenv, load_dotenv

# First, load the environment variables from the .env file
load_dotenv(
    find_dotenv(
        # Use the current working directory from where the command is run
        usecwd=True,
    )
)

# Detect if the code is running in a CI environment
# See: https://stackoverflow.com/a/75223617
IS_CI = getenv("CI", "").lower() == "true"
