
# Universal LLM integration

A streamlit llm app that allow developer to test various llm model from different providers


## Deployment

The app can easily be run locally or deployed on a virtual machine with a single `docker compose` command

```bash
  docker compose up
```


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`HUGGINGFACE_API_KEY`


## Documentation

Ollama has been deployed with docker, so you need to connect to the runned container to add models using the following commang


```bash
  ollama pull model_name
```

