import requests


def pixabay_api_request(
    api_key,
    query,
    lang="en",
    image_type="all",
    orientation="all",
    category=None,
    min_width=0,
    min_height=0,
    colors=None,
    editors_choice=False,
    safesearch=False,
    order="popular",
    page=1,
    per_page=20,
    callback=None,
    pretty=False,
):
    # Pixabay API endpoint
    api_url = "https://pixabay.com/api/"

    # Construct parameters for the request
    params = {
        "key": api_key,
        "q": query,
        "lang": lang,
        "image_type": image_type,
        "orientation": orientation,
        "category": category,
        "min_width": min_width,
        "min_height": min_height,
        "colors": colors,
        "editors_choice": "true" if editors_choice else "false",
        "safesearch": "true" if safesearch else "false",
        "order": order,
        "page": page,
        "per_page": per_page,
        "callback": callback,
        "pretty": "true" if pretty else "false",
    }

    # Make the API request
    response = requests.get(api_url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse and return the JSON response
        return response.json()
    else:
        # Print the error message if the request was unsuccessful
        print(f"Error: {response.status_code}")
        return None
