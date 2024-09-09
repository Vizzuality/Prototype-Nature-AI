
def get_evaluation_parameters():

    params = {}

    params['Nreplicates'] = 1
    params['llm_choice'] = "GoogleGenerativeAI"
    params['model'] = "gemini-1.0-pro" #llama3.1" #model choice if using a local Ollama model
    params['temperature'] = 1
    params['persist_directory'] = '../training/wikipedia'

    return params

def get_app_parameters():

    params = {}

    params['Nreplicates'] = 1
    params['llm_choice'] = "ChatGPT"
    params['model'] = "gpt-4o" #llama3.1" #model choice if using a local Ollama model
    params['temperature'] = 1
    params['persist_directory'] = 'training/wikipedia'

    return params
