# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that provides concise and accurate information. Always provide the hierachical information with more than 3 levels in depth, and less than 5 points in each level, including the top level, which means onces each level has more than 5 points, you should sort out to combine some points into the next level. Use markdown list formatting for the response, so the hierarchy structure can be rendered on the page. Response should only be the content, no other text."},
        {"role": "user", "content": "hello welcome to icliniq com your egg is fine there are no significant abnormalities seen in egg electrocardiogram machine generated report mentioned at the top is frequently inaccurate so not to get panic with it now in cardiac pain the pain usually occurs on exertion and relieves with rest this is very classical of cardiac pain if it persists for a few minutes and does not last round the clock more than 2 days now if the pain does not have such characteristic then it may either be related to cervical spondylosis or gastric reflux if you have any bloating burping heartburn or upper abdominal pain then it may be related to gastric reflux or if this pain increases on neck movements associated with tingling numbness sensation in arms then it may be related to cervical pain however considering the risk factors like do diabetes mellitus hen hypertension it is better to rule out the possibility of cardiac pain so you should undergo echo and if normal then treadmill test"},
    ],
    stream=False
)

print(response)
print(response.choices[0].message.content)