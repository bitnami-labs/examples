# Website with Node.js, Express, and Bootstrap

A simple website to provide information on container and chart solutions. It uses `app.js` as main entrypoint, and a `views` directory to include the static assets.

## Prerequisites

- Node.js and npm (v18.04 or higher)
- Docker

## Execution

### Without Docker

1.  **Install dependencies:**
    ```bash
    npm install
    ```
2.  **Run the application:**
    ```bash
    node app.js
    ```

### Using Docker

This section uses a multi-stage Dockerfile to create a lightweight and secure image. It first builds the application dependencies using a Bitnami Node.js image and then copies the final, lean application into a Bitnami Node.js Minimal image for the runtime.

1.  **Build the Docker image:**
    ```bash
    docker build -t nodejs-example .
    ```
2.  **Run the container:**
    Execute the command below. It maps port `8080` on your host machine to port `8080` in the container and names the container `nodejs-example`.
    ```bash
    docker run --rm -it -p 8080:8080 --name nodejs-example nodejs-example
    ```

---

## Access the Website

Open your web browser and go to `http://localhost:8080`.

Explore the different pages by navigating to the following endpoints:

* `/`
* `/containers`
* `/charts`


## Usage

You can now navigate throw the website and interact with the system at `/`, `/containers`,  and `/charts` endpoints 🚀.
