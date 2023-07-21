from Markov import MarkovModel

# Load the text data
with open('stud.txt', 'r') as file1, open('sign.txt', 'r') as file2:
    text1 = file1.read()
    text2 = file2.read()

# Combine the texts
text = text1 + text2

# Create and train the model
model = MarkovModel()
model.train(text)

# Generate and print new text
new_text = model.generate_text(2000)
print(new_text)

# Save the new text to a file
with open('Readme.txt', 'w') as file:
    file.write(new_text)
