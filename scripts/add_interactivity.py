"""Inject copy buttons, knowledge checks, and try-it-yourself sections into all pages."""
from pathlib import Path
import re

PAGES = Path(__file__).resolve().parents[1] / "app" / "pages"

QUIZZES = {
"python.html": [
  {"q": "What does <code>list.append(x)</code> do?",
   "opts": ["Adds x to the start of the list", "Adds x to the end of the list", "Returns a new list containing x"],
   "correct": 1, "exp": "<code>append</code> mutates the list in-place, adding the element at the end. Use <code>insert(0, x)</code> to prepend."},
  {"q": "Which of these is a valid list comprehension?",
   "opts": ["<code>[x*2 for x in range(5)]</code>", "<code>{x*2 for x in range(5)}</code>", "<code>(x*2 for x in range(5))</code>"],
   "correct": 0, "exp": "Square brackets make a list comprehension. Curly braces make a set comprehension; parentheses make a generator expression."},
  {"q": "What does <code>*args</code> capture in a function signature?",
   "opts": ["Keyword arguments as a dict", "All positional arguments as a tuple", "Both positional and keyword arguments"],
   "correct": 1, "exp": "<code>*args</code> collects extra positional arguments into a tuple. Use <code>**kwargs</code> for keyword arguments."},
],
"ml_basics.html": [
  {"q": "Why should you only fit a <code>StandardScaler</code> on training data?",
   "opts": ["Scalers perform better on larger datasets", "Fitting on test data leaks test statistics into training (data leakage)", "Training data is always cleaner"],
   "correct": 1, "exp": "Fitting on the full dataset means test-set statistics influence training — an optimistic bias called data leakage."},
  {"q": "What does <code>A @ B</code> compute in NumPy?",
   "opts": ["Element-wise multiplication", "Matrix dot product", "Transpose of A multiplied by B"],
   "correct": 1, "exp": "<code>@</code> is the matrix multiplication operator (PEP 465). <code>A * B</code> is element-wise multiplication."},
  {"q": "What does <code>df.groupby('col').mean()</code> return?",
   "opts": ["A single mean for the whole DataFrame", "The mean of each group defined by the unique values in col", "The count of each unique value in col"],
   "correct": 1, "exp": "<code>groupby</code> splits the DataFrame by the column values; <code>.mean()</code> then aggregates each group separately."},
],
"supervised_learning.html": [
  {"q": "What is the key difference between classification and regression?",
   "opts": ["Classification predicts continuous values; regression predicts categories", "They are the same — both predict numeric outputs", "Classification predicts discrete categories; regression predicts continuous values"],
   "correct": 2, "exp": "Regression outputs a number on a continuous scale (e.g. price). Classification outputs a class label (e.g. spam / not-spam)."},
  {"q": "In KNN, what happens as k approaches the full training set size?",
   "opts": ["The model memorises every training point", "The model predicts the majority class for every input", "Accuracy always improves with larger k"],
   "correct": 1, "exp": "With k equal to the full training set, every prediction is the global majority class — maximum bias, minimum variance."},
  {"q": "Which metric applies to classification but NOT regression?",
   "opts": ["Mean Squared Error", "Accuracy", "R² Score"],
   "correct": 1, "exp": "Accuracy counts correctly classified samples — it only makes sense for discrete labels. MSE and R² apply to continuous predictions."},
],
"unsupervised_learning.html": [
  {"q": "What quantity does K-Means minimise?",
   "opts": ["Number of cluster centres", "Within-cluster sum of squared distances (inertia)", "Between-cluster Euclidean distance"],
   "correct": 1, "exp": "K-Means iteratively reassigns points to the nearest centroid and moves centroids to minimise total squared intra-cluster distance."},
  {"q": "What does PCA's first principal component represent?",
   "opts": ["The feature with the highest individual variance", "The direction of maximum variance across the full dataset", "The mean vector of the dataset"],
   "correct": 1, "exp": "PCA finds orthogonal axes that maximise captured variance. PC1 explains more variance than any other single direction."},
  {"q": "What is the elbow method used for in K-Means?",
   "opts": ["Choosing the learning rate", "Choosing the number of clusters k", "Detecting outliers before clustering"],
   "correct": 1, "exp": "Plot inertia vs k. The elbow — where inertia stops falling quickly — is a good candidate for k."},
],
"deep_learning.html": [
  {"q": "What does <code>ReLU</code> return for a negative input?",
   "opts": ["The input unchanged", "Zero", "A small negative value (leaky ReLU)"],
   "correct": 1, "exp": "ReLU(x) = max(0, x). Negative values are clipped to zero, which removes them from the gradient path."},
  {"q": "What does <code>optimizer.zero_grad()</code> do?",
   "opts": ["Initialises model weights to zero", "Clears gradients accumulated from the previous step", "Resets the learning rate scheduler"],
   "correct": 1, "exp": "PyTorch accumulates gradients by default. Calling <code>zero_grad()</code> before <code>backward()</code> ensures each step uses only the current batch's gradients."},
  {"q": "What does Dropout do during inference (<code>model.eval()</code>)?",
   "opts": ["Continues randomly dropping neurons", "Becomes a no-op — all neurons stay active", "Scales outputs by the dropout probability"],
   "correct": 1, "exp": "Dropout is disabled in eval mode so the full network is used for every prediction. The expected value is preserved by the scale-at-train-time approach."},
],
"computer_vision.html": [
  {"q": "What does a MaxPool layer do?",
   "opts": ["Averages all values in the pooling window", "Takes the maximum value in each pooling window", "Applies a learnable filter to the input"],
   "correct": 1, "exp": "MaxPooling downsamples the feature map by retaining the strongest activation per region, providing translation invariance."},
  {"q": "Why do CNNs outperform fully-connected networks on images?",
   "opts": ["CNNs share weights across spatial locations, drastically reducing parameters", "CNNs have more neurons per layer", "CNNs always train faster on GPUs"],
   "correct": 0, "exp": "Weight sharing means a filter detecting an edge works anywhere in the image — a massive parameter reduction vs a dense layer per pixel."},
  {"q": "What is the purpose of padding in a convolution?",
   "opts": ["To speed up the computation", "To preserve the spatial dimensions of the feature map", "To reduce the number of filters needed"],
   "correct": 1, "exp": "Without padding, each convolution shrinks the output. Same padding keeps height and width constant, which is important for deep architectures."},
],
"natural_language_processing.html": [
  {"q": "What does Word2Vec learn?",
   "opts": ["Exact dictionary definitions of words", "Dense vectors where semantically similar words have similar representations", "A bag-of-words frequency matrix"],
   "correct": 1, "exp": "Word2Vec trains a shallow network to predict context words. The resulting embeddings encode meaning: king - man + woman ~ queen."},
  {"q": "What is a bigram?",
   "opts": ["A word with exactly two syllables", "Two consecutive tokens treated as a single unit", "A pair of antonyms"],
   "correct": 1, "exp": "An n-gram is a contiguous sequence of n tokens. A bigram (n=2) captures common phrases like 'New York' or 'machine learning'."},
  {"q": "Why is PMI useful for finding collocations?",
   "opts": ["It counts raw co-occurrence frequencies", "It measures how much more often two words co-occur than expected by chance", "It removes stopwords before counting"],
   "correct": 1, "exp": "Raw counts favour frequent words. PMI normalises by individual frequencies, surfacing genuinely meaningful associations like 'peanut butter'."},
],
"time_series.html": [
  {"q": "What does stationarity mean in a time series?",
   "opts": ["Statistical properties (mean, variance, autocorrelation) are constant over time", "The series has no seasonal component", "The series values are always positive"],
   "correct": 0, "exp": "A stationary series has constant mean, variance, and autocorrelation structure. ARIMA requires stationarity; differencing is the standard fix."},
  {"q": "What does the I in ARIMA stand for?",
   "opts": ["Integrated", "Intercept", "Independent"],
   "correct": 0, "exp": "I(d) means the series is differenced d times to achieve stationarity. AR = autoregressive, MA = moving average."},
  {"q": "Why is differencing applied before fitting ARIMA?",
   "opts": ["To add a seasonal component", "To make a non-stationary series stationary", "To smooth short-term fluctuations"],
   "correct": 1, "exp": "Differencing removes trends by subtracting consecutive values, turning a random walk into stationary noise that ARIMA can model."},
],
"recommendation_system.html": [
  {"q": "What does cosine similarity measure?",
   "opts": ["Euclidean distance between two vectors", "The cosine of the angle between two vectors", "The dot product scaled by vector length squared"],
   "correct": 1, "exp": "Cosine similarity = (A.B)/(|A||B|). It measures directional similarity regardless of magnitude — ideal for sparse user-item vectors."},
  {"q": "What is the cold start problem?",
   "opts": ["The recommendation engine is slow to initialise", "New users or items have no interaction history for recommendations", "The model overfits to the most popular items"],
   "correct": 1, "exp": "Without ratings or behaviour data, collaborative filtering has nothing to work from. Solutions: ask preferences on signup, or fall back to content-based filtering."},
  {"q": "What is the key difference between content-based and collaborative filtering?",
   "opts": ["Content-based uses item/user features; collaborative uses interaction patterns", "Content-based is always more accurate", "Collaborative filtering never suffers from cold start"],
   "correct": 0, "exp": "Content-based: recommend items similar to ones you liked (needs feature data). Collaborative: recommend what similar users liked (needs interaction history)."},
],
}

EXERCISES = {
"python.html": [
  {"title": "List comprehension challenge",
   "body": "Write a single list comprehension that returns the squares of all odd numbers from 1 to 20.",
   "hint": "<code>[x**2 for x in range(1, 21) if x % 2 != 0]</code><br>Expected: [1, 9, 25, 49, 81, 121, 169, 225, 289, 361]"},
  {"title": "Function with *args",
   "body": "Write a function <code>running_total(*numbers)</code> that returns a list of cumulative sums.<br><code>running_total(1, 2, 3, 4)</code> should return <code>[1, 3, 6, 10]</code>.",
   "hint": "<pre style='margin:0'><code>def running_total(*numbers):\n    total = 0\n    result = []\n    for n in numbers:\n        total += n\n        result.append(total)\n    return result</code></pre>"},
],
"ml_basics.html": [
  {"title": "Row-normalise a matrix",
   "body": "Given <code>X = np.array([[3,4],[1,0],[0,2]])</code>, divide each row by its L2 norm so every row has unit length.",
   "hint": "<code>norms = np.linalg.norm(X, axis=1, keepdims=True)</code><br><code>X_norm = X / norms</code><br>Check: <code>np.linalg.norm(X_norm, axis=1)</code> should be all 1.0"},
  {"title": "GroupBy aggregation",
   "body": "Create a DataFrame with <code>year</code> (2022/2023), <code>product</code> (A/B), and <code>sales</code> columns. Find total and mean sales per year.",
   "hint": "<code>df.groupby('year')['sales'].agg(['sum', 'mean'])</code>"},
],
"supervised_learning.html": [
  {"title": "Tune k in KNN",
   "body": "Train a KNN classifier for k = 1, 3, 5, 7, 9, 11 on the Iris dataset (80/20 split). Print test accuracy for each k. Which gives the best generalisation?",
   "hint": "<pre style='margin:0'><code>from sklearn.neighbors import KNeighborsClassifier\nfor k in [1,3,5,7,9,11]:\n    acc = KNeighborsClassifier(k).fit(X_train,y_train).score(X_test,y_test)\n    print(f'k={k}: {acc:.2%}')</code></pre>"},
],
"deep_learning.html": [
  {"title": "Add a hidden layer",
   "body": "Extend IrisNet with a third hidden layer (16 → 8 neurons) before the final output layer. Does validation accuracy improve over 100 epochs?",
   "hint": "Insert <code>nn.Linear(16, 8), nn.ReLU()</code> after the 32→16 layer. Update the final layer to <code>nn.Linear(8, 3)</code>."},
  {"title": "Compare optimisers",
   "body": "Replace <code>optim.Adam</code> with <code>optim.SGD(lr=0.1, momentum=0.9)</code> and compare loss curves over 100 epochs.",
   "hint": "SGD with momentum needs a higher lr than Adam but often converges to sharper minima. You should see noisier early epochs but comparable final accuracy."},
],
}

COPY_CSS = """
        .copy-btn {
            position: absolute; top: 8px; right: 8px;
            background: rgba(255,255,255,0.08); color: #aaa;
            border: 1px solid rgba(255,255,255,0.2); border-radius: 4px;
            padding: 2px 10px; font-size: 11px; font-family: sans-serif;
            cursor: pointer; transition: all 0.15s; z-index: 10;
        }
        .copy-btn:hover { background: rgba(255,255,255,0.22); color: #fff; }
        .copy-btn.copied { background: #22c55e22; color: #86efac; border-color: #22c55e44; }
        details.try-it > summary { list-style: none; }
        details.try-it > summary::-webkit-details-marker { display: none; }"""

COPY_JS = """
    <script>
        document.querySelectorAll('pre').forEach(function(pre) {
            var btn = document.createElement('button');
            btn.textContent = 'Copy';
            btn.className = 'copy-btn';
            pre.style.position = 'relative';
            pre.appendChild(btn);
            btn.addEventListener('click', function() {
                var code = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
                navigator.clipboard.writeText(code).then(function() {
                    btn.textContent = 'Copied!';
                    btn.classList.add('copied');
                    setTimeout(function() { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
                }).catch(function() { btn.textContent = 'Error'; setTimeout(function() { btn.textContent = 'Copy'; }, 2000); });
            });
        });
    </script>"""


def quiz_section(questions):
    items = []
    for i, q in enumerate(questions, 1):
        opts = "".join(
            f'<label class="flex items-start gap-2 cursor-pointer hover:bg-gray-50 p-2 rounded-lg">'
            f'<input type="radio" name="q{i}" value="{j}" class="mt-1 accent-indigo-600 flex-shrink-0">'
            f'<span>{o}</span></label>'
            for j, o in enumerate(q["opts"])
        )
        items.append(
            f'<div class="bg-white border border-gray-200 rounded-xl p-5">'
            f'<p class="font-semibold text-gray-800 mb-3">{i}. {q["q"]}</p>'
            f'<div class="space-y-1">{opts}</div>'
            f'<div id="q{i}-feedback" class="hidden mt-3 p-3 rounded-lg text-sm"></div>'
            f'</div>'
        )

    answers_js = (
        "["
        + ",".join(
            '{correct:' + str(q["correct"]) + ',exp:"' + q["exp"].replace('"', "'") + '"}'
            for q in questions
        )
        + "]"
    )

    return (
        '\n\t\t\t\t\t<!-- Knowledge Check -->\n'
        '\t\t\t\t\t<section id="quiz" class="content-section">\n'
        '\t\t\t\t\t\t<h2><i class="fa-solid fa-circle-question mr-3 text-indigo-500"></i> Knowledge Check</h2>\n'
        '\t\t\t\t\t\t<p>Test your understanding — pick the best answer, then click Check.</p>\n'
        '\t\t\t\t\t\t<div class="space-y-4 mt-4">\n'
        + "".join(f'\t\t\t\t\t\t\t{item}\n' for item in items)
        + '\t\t\t\t\t\t</div>\n'
        '\t\t\t\t\t\t<div class="flex items-center gap-4 mt-5">\n'
        '\t\t\t\t\t\t\t<button onclick="checkQuiz()" class="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium transition">Check Answers</button>\n'
        '\t\t\t\t\t\t\t<button onclick="resetQuiz()" class="px-4 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50 text-sm transition">Reset</button>\n'
        '\t\t\t\t\t\t</div>\n'
        '\t\t\t\t\t\t<div id="quiz-score" class="hidden mt-4 p-4 bg-indigo-50 border border-indigo-200 rounded-xl text-indigo-800 font-semibold text-center text-lg"></div>\n'
        '\t\t\t\t\t</section>\n'
        f'\t\t\t\t\t<script>\n'
        f'\t\t\t\t\t\tvar QUIZ_DATA={answers_js};\n'
        '\t\t\t\t\t\tfunction checkQuiz(){var score=0;QUIZ_DATA.forEach(function(q,i){var sel=document.querySelector(\'input[name="q\'+(i+1)+\'"]:checked\');var fb=document.getElementById(\'q\'+(i+1)+\'-feedback\');if(!sel){fb.innerHTML=\'Please select an answer.\';fb.className=\'mt-3 p-3 rounded-lg text-sm bg-yellow-50 text-yellow-800 border border-yellow-200\';fb.classList.remove(\'hidden\');return;}var val=parseInt(sel.value);if(val===q.correct){fb.innerHTML=\'<strong>Correct!</strong> \'+q.exp;fb.className=\'mt-3 p-3 rounded-lg text-sm bg-green-50 text-green-800 border border-green-200\';score++;}else{fb.innerHTML=\'<strong>Not quite.</strong> \'+q.exp;fb.className=\'mt-3 p-3 rounded-lg text-sm bg-red-50 text-red-800 border border-red-200\';}fb.classList.remove(\'hidden\');});var sd=document.getElementById(\'quiz-score\');sd.textContent=\'Score: \'+score+\' / \'+QUIZ_DATA.length+\'  (\'+Math.round(score/QUIZ_DATA.length*100)+\'%)\';sd.classList.remove(\'hidden\');}\n'
        '\t\t\t\t\t\tfunction resetQuiz(){document.querySelectorAll(\'input[type=radio]\').forEach(function(r){r.checked=false;});QUIZ_DATA.forEach(function(_,i){var fb=document.getElementById(\'q\'+(i+1)+\'-feedback\');fb.className=\'hidden\';});document.getElementById(\'quiz-score\').classList.add(\'hidden\');}\n'
        '\t\t\t\t\t</script>'
    )


def exercises_html(exercises):
    blocks = []
    for ex in exercises:
        blocks.append(
            '\t\t\t\t\t<details class="try-it bg-white border border-indigo-100 rounded-xl overflow-hidden mt-4">\n'
            f'\t\t\t\t\t\t<summary class="px-5 py-3 cursor-pointer font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 transition flex items-center gap-2 select-none">'
            f'<i class="fa-solid fa-laptop-code mr-1"></i> Try it yourself &mdash; {ex["title"]}'
            f'</summary>\n'
            f'\t\t\t\t\t\t<div class="p-5">'
            f'<p class="text-gray-700 mb-3">{ex["body"]}</p>'
            f'<details class="mt-3"><summary class="text-sm text-indigo-500 cursor-pointer hover:text-indigo-700 font-medium">Show hint</summary>'
            f'<div class="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-700">{ex["hint"]}</div>'
            f'</details></div>\n'
            '\t\t\t\t\t</details>'
        )
    return '\n\t\t\t\t\t<!-- Try it yourself -->\n' + '\n'.join(blocks) + '\n'


def patch(page: Path):
    text = page.read_text(encoding="utf-8")
    changed = False

    # 1. CSS in <style>
    if "copy-btn" not in text:
        text = text.replace("    </style>", COPY_CSS + "\n    </style>", 1)
        changed = True

    # 2. Exercises + quiz before </main>
    if "quiz" not in text and page.name in QUIZZES:
        inject = ""
        if page.name in EXERCISES:
            inject += exercises_html(EXERCISES[page.name])
        inject += quiz_section(QUIZZES[page.name])
        # Try several indentation patterns
        for closing in ["\n\t\t\t</main>", "\n\t\t</main>", "\n\t</main>", "\n</main>"]:
            if closing in text:
                text = text.replace(closing, inject + closing, 1)
                changed = True
                break

    # 3. Sidebar link
    if 'href="#quiz"' not in text and page.name in QUIZZES:
        # Find last sidebar-link anchor and append after it
        pattern = r'(<a href="#[^"]*" class="sidebar-link[^"]*"[^>]*>[^<]*</a>\n)(\s*</nav>)'

        def repl(m):
            return m.group(1) + m.group(2).replace(
                '</nav>',
                '\t\t\t\t\t\t<a href="#quiz" class="sidebar-link" onclick="setActive(this)">Knowledge Check</a>\n\t\t\t\t\t</nav>'
            )
        new = re.sub(pattern, repl, text)
        if new != text:
            text = new
            changed = True

    # 4. Copy-button JS before </body>
    if "navigator.clipboard" not in text:
        text = text.replace("</body>", COPY_JS + "\n</body>", 1)
        changed = True

    if changed:
        page.write_text(text, encoding="utf-8")
    return changed


skip = {"machine_learning.html"}
patched, skipped = [], []
for page in sorted(PAGES.glob("*.html")):
    if page.name in skip:
        skipped.append(page.name)
        continue
    if patch(page):
        patched.append(page.name)
    else:
        skipped.append(page.name + " (no change)")

print("Patched :", patched)
print("Skipped :", skipped)
