from openai import OpenAI

client = OpenAI(api_key="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")

models = client.models.list()

for model in models.data:
    print(model.id)
