function trackEntry(text) {
    let text = document.getElementById('entry').value

    if (text == "hello") {
        alert("hello")
    }
}

function runPrediction() {
    let words = []
    // words entered into entry field
    let entry = document.getElementById('entry')
    // on keypress of space, add word to array
    // if (entry) {}
    entry.addEventListener("onkeydown", () => words.push(word))
    if (words.length == 3) {
        // run the model
    }
}