
# FloatStream.ai

FloatStream.ai is a document management and querying system that uses vector embeddings for efficient similarity search.

## Steps to run the app
1. source ./venv/bin/activate
2. chmod +x setup_and_run.sh 
3. ./setup_and_run.sh

## Setup

1. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

   # Or

   source ./venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt

   # Or

   chmod +x setup_and_run.sh 
   ./setup_and_run.sh
   ```

4. Set up the configuration:
   - The `config` directory contains JSON files for different environments.
   - Set the `ENVIRONMENT` variable in the `.env` file to switch between configurations.


2. The API will be available at `http://localhost:8000`

3. You can access the Front-end at ` http://localhost:8080`

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