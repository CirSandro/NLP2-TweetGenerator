
import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import nltk
from nltk.tokenize import word_tokenize
import re
import string

# Fonction pour nettoyer le texte
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub('\n', '', text)
    return text

# Fonction pour extraire le sujet principal de la phrase
def extract_subject(sentence):
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    subjects = [word for word, tag in tagged if tag.startswith('NN')]
    return subjects

# Fonction pour générer les commentaires pour un tweet donné
def generate_comments(tweet, num_comments):
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    messages = [
        {"role": "system", "content": f"Generate exactly {num_comments} distinct and independent Twitter comments representing {num_comments} user reactions. Each comment should reflect the viewpoint of the tweet in question, adapt to the written language, stay within the 280-character limit, and appear realistic for a Twitter comment where each comment must correspond to a reaction that a user could have when reading this tweet. Returns comments in this form: '1. first comment\n\n2. Second Comment\n\n....'"},
        {"role": "user", "content": tweet},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    liste = output[0]['generated_text'].split('\n')
    comments = []
    for line in liste:
        if line.strip() != "":
            separator_index = line.find(". ")
            if separator_index != -1:
                index = separator_index + 2
                comments.append(line[index:].replace('"', ''))
            else:
                comments.append(line.replace('"', ''))
    comments = comments[:num_comments]

    return comments

# Chargement des données d'entraînement
train = pd.read_csv('data/tweet_train.csv')
train.dropna(inplace=True)
train['original_text'] = train['text'].copy()
train['original_selected_text'] = train['selected_text'].copy()
train['text'] = train['text'].apply(clean_text)
train['selected_text'] = train['selected_text'].apply(clean_text)

# Interface utilisateur avec Streamlit
def main():    
    st.title('Analyse des Tweets')

    # Affichage des données d'entraînement
    # st.subheader("Données d'entraînement")
    # st.write(train)

    # Sélection d'un tweet aléatoire
    #sample_tweet = st.selectbox("Sélectionner un tweet aléatoire", train['original_text'])
    selected_tweet = st.selectbox('Sélectionner un tweet', train['original_text'])

    # Affichage du tweet sélectionné
    st.subheader("Tweet sélectionné")
    st.write(selected_tweet)

    # Génération de commentaires
    st.subheader("Génération de commentaires")
    nbr_comment = st.number_input("Combien de commentaires générer?", min_value=1, max_value=10, value=1)

    if st.button("Générer les commentaires"):
        # Nettoyage du tweet sélectionné
        cleaned_tweet = clean_text(selected_tweet)

        # Génération des commentaires
        comments = generate_comments(cleaned_tweet, nbr_comment)

        # Affichage des commentaires générés
        for i, comment in enumerate(comments):
            st.write(f"{i+1}. {comment}")

    # Créer un nouveau tweet avec un nombre spécifié de commentaires
    st.sidebar.header('Nouveau tweet')
    new_tweet = st.sidebar.text_area('Nouveau tweet')
    nbr_comment = st.sidebar.number_input('Nombre de commentaires', min_value=1, max_value=15, value=5)

    if st.sidebar.button('Générer pour le nouveau tweet'):
        generated_comments = generate_comments(new_tweet, nbr_comment)
        st.subheader('Nouveaux commentaires générés')
        for i, comment in enumerate(generated_comments):
            st.write(f"{i+1}. {comment}")


# Exécuter l'interface utilisateur
if __name__ == '__main__':
    main()
