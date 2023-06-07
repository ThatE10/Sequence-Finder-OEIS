import numpy as np
from flask import Flask, render_template, request
from OEIS_access import get_results
from IRLS import hm_irls
import re

app = Flask(__name__)


# define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    # if the user submitted a search query
    if request.method == 'POST':
        # get the search query from the form
        query = request.form.get('query')
        # perform the search
        try:
            calculated_query, results = search(query)
            return render_template('index.html', query=calculated_query, results=results)
        except Exception as e:
            print("your exception:", e)
            return render_template('index.html', query=query, error=e)
        # pass the results to the template

    # if the user loaded the page without submitting a search query
    else:
        return render_template('index.html')


# define a search function that returns a list of result items
def search(query):
    print("hello world!!")
    # parse query and clean it.
    string = query[:]
    elements = string.split(',')  # Split the string by commas
    elements = [element for element in elements if element]  # Remove empty elements

    elements = [element for element in elements if is_number(element) or element == "?"]
    # Keep elements that are valid numbers or equal to "?"
    # using list comprehension

    string = ','.join(elements)  # Join the elements with commas
    if (string != query):
        print(string)
        print(query)
        print("invalid!!")
        raise Exception("Your sequence:",query," is invalid")


    clean_query_int = string.split(',')

    if (len(clean_query_int) < 5):
        raise Exception("You have provided too little information at least 5 entries")
    clean_query_int = [float("nan") if num == "?" else float(num) for num in clean_query_int]

    [calculated_sequence, stats] = hm_irls(clean_query_int, rank_estimate=2, max_iter=400, type_mean='geometric',tol=1e-10)
    print("hello world!")
    calculated_sequence = np.round(calculated_sequence, 8)
    print(calculated_sequence)
    results = get_results(calculated_sequence)

    return (calculated_sequence, results)

def is_number(element):
    try:
        float(element)  # Try converting the element to float
        return True     # If successful, it is a valid number
    except ValueError:
        return False    # If ValueError is raised, it is not a valid number

if __name__ == '__main__':
    app.run()
