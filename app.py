from flask import Flask,render_template,request
from transformers import GPT2LMHeadModel, GPT2Tokenizer,pipeline
from happytransformer import HappyTextToText, TTSettings

app = Flask(__name__)

input=""
@app.route('/')
def testing():
    return render_template("test.html")


def generateText(input):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)
    tokenizer.decode(tokenizer.eos_token_id)
    input_ids = tokenizer.encode(input, return_tensors='pt')
    output = model.generate(input_ids, max_length=200, num_beams=5,no_repeat_ngram_size=2, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def CorrectGrammer(input):
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    args = TTSettings(num_beams=5, min_length=1)
    result = happy_tt.generate_text(input, args=args)
    return result.text


def TextSummarization(input):
    summarizer = pipeline("summarization")
    summarytext = summarizer(input)
    return summarytext[0]['summary_text']


def Sentiment(input):
    sentiment = pipeline("sentiment-analysis")
    return sentiment(input)[0]['label']

@app.route('/input', methods=['POST', 'GET'])
def input():
    input = request.form['input']
    if request.method == 'POST':
        if request.form['output'] == 'WordCount':
            return render_template('test.html', input = input , output=len(input.split()))
        
        if request.form['output'] == 'Sentiment':
            return render_template('test.html', input=input,  output=Sentiment(input))

        if request.form['output'] == 'TextSummarization':
            return render_template('test.html', input=input, output=TextSummarization(input))

        if request.form['output'] == 'CorrectGrammer':
            return render_template('test.html', input=input, output=CorrectGrammer(input))

        if request.form['output'] == 'GenerateText':
            return render_template('test.html', output=generateText(input))

    return render_template('test.html', output=input)
app.run()