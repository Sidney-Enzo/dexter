import requests
import asyncio
import python_weather

def get_location() -> dict[str]:
    response = requests.get('https://ipinfo.io')
    return response.json()

def get_weather(city: str) -> str:
    async def fetch_weather():
        async with python_weather.Client() as client:
            weather = await client.get(city)
            return (f'Today the weather is {weather.description}, '
                    f'temperatures in {weather.temperature} °C, '
                    f'the wind speed is {weather.wind_speed}km/h '
                    f'and the air humidity is in {weather.humidity}%.')

    return asyncio.run(fetch_weather())