# FloatStream.io

FloatStream.io is a document management and querying system that uses vector embeddings for efficient similarity search.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/floatstream-io.git
   cd floatstream-io
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the configuration:
   - The `config` directory contains JSON files for different environments.
   - Set the `ENVIRONMENT` variable in the `.env` file to switch between configurations.

## Running the Application

1. Start the FastAPI server:
   ```
   uvicorn app:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. You can access the API documentation at `http://localhost:8000/docs`

## Usage

- Use the `/index` endpoint to upload and index documents.
- Use the `/query` endpoint to search the indexed documents.
- Use the `/delete_all` endpoint to remove all indexed documents.
- Other endpoints provide system information and statistics.

## Configuration

You can modify the `config/development.json` and `config/production.json` files to adjust various parameters such as:

- Model names
- Chunk sizes
- Search parameters
- File paths

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.