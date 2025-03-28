import gradio as gr
import requests
import json
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from Helper import ( load_chroma_databases, retrieve_relevant_chunks,retrieve_relevant_chunks_Pass_Chapters_To_BestChunks,Chapters_Intros)

#  CONFIGURATION  #


model_name = "sentence-transformers/all-MiniLM-L6-v2"
Model = HuggingFaceEmbeddings(model_name=model_name)

#  GRADIO FUNCTIONS  #
def get_chunk_params(answer_length):
    params = {"Very Short": (800, 150, 2),"Short": (800, 150, 3),"Long": (800, 150, 4),"Very Long": (800, 150, 5)}    
    return params.get(answer_length, (1000, 100,6))


def retrieve_text(query,length_input):
    selected_chapters_UnProcessed=retrieve_relevant_chunks_Pass_Chapters_To_BestChunks('./ChromaDB_Query/Intro_SelectChapters',Chapters_Intros,Model,query,5)
    selected_chapters = [doc.metadata['chapter'] for doc in selected_chapters_UnProcessed]

    chunk_size, chunk_overlap, top_k_per_chapter =get_chunk_params(length_input)
    top_k = 10#((top_k_per_chapter) * len(selected_chapters)) // 2
    
    SelectedChapters_text = load_chroma_databases(model_name,"text",chunk_size,chunk_overlap,selected_chapters=selected_chapters)
    Docs_Score_Text = retrieve_relevant_chunks(query, SelectedChapters_text, top_k_per_chapter)
    if(len(Docs_Score_Text)>10):
        Docs_Score_Text=retrieve_relevant_chunks_Pass_Chapters_To_BestChunks('ChromaDB_Query/Text',Docs_Score_Text,Model,query,top_k)    
    Retrieved_Text = " ".join([doc.page_content for doc in Docs_Score_Text])


    #SelectedChapters_Table= load_chroma_databases(model_name,"table",chunk_size,chunk_overlap,selected_chapters=selected_chapters)
    #Docs_Score_Table = retrieve_relevant_chunks(query, SelectedChapters_Table, top_k_per_chapter)
    #if(len(Docs_Score_Table)>10):
    #    Docs_Score_Table = retrieve_relevant_chunks_Pass_Chapters_To_BestChunks('./ChromaDB_Query/Table',Docs_Score_Table,Model,query,top_k)
    #Retrieved_Table = " ".join([doc.page_content for doc in Docs_Score_Table])
    Retrieved_Table="NO TABLE"

    LLm_Generated_Answer=generate_llama_response(Retrieved_Text,Retrieved_Table, query)

    return LLm_Generated_Answer

def generate_llama_response(RetrievedText_Text,RetrievedText_Table, query):
    if not RetrievedText_Text.strip():
        return " No retrieved text available for answering."

    payload = {
        "model": "llama3",
        "prompt": f"I will provide you content. Please use that content only to answer my query. Text: {RetrievedText_Text}, Table:{RetrievedText_Table} Query: {query}",
    }
    
    try:
        url = 'http://localhost:11434/api/generate'
        response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            list_dict_words = []
            for each_word in response.text.split("\n"):
                try:
                    data = json.loads(each_word)
                except:
                    pass
                list_dict_words.append(data)
            
            llama_response = " ".join([word['response'] for word in list_dict_words if isinstance(word, dict)])
            return llama_response if llama_response else " No meaningful response from LLama3."

        return f" API Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f" Failed to connect to Llama API: {str(e)}"
    
#  GRADIO INTERFACE  #
with gr.Blocks() as demo:
    with gr.Tab("Ask a Question"):
        chapter_options = [str(i) for i in range(1, 45)]  

        length_input = gr.Dropdown(choices=["Very Short", "Short", "Long", "Very Long"],label="Select Answer Length",value="Very Long", interactive=True)
        query_input = gr.Textbox(label="Enter your Query")
        
        retrieve_button = gr.Button("Retrieve Answer")
        
        retrieved_output = gr.Textbox(label="Retrieved Chunks", interactive=False)
        retrieve_button.click(
            retrieve_text, 
            inputs=[query_input,length_input], 
            outputs=[retrieved_output]  
        )


#  RUN GRADIO  #
demo.launch()


#Tuning

# Choosing LLM
# LLM model prompt tune
# Table


#query = "what materials are acceptable for fireblocking in residential construction?"
