import os
import pickle
import re
import pymupdf  
import shutil
from langchain.schema import Document 
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json

#   Preprocessing.py helper functions   #
def clean_pdf_text(text):
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text) 
    text = re.sub(r"(\w)(\n)(\w)", r"\1 \3", text) 
    text = re.sub(r"\s*\n\s*", " ", text) 
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_tables(page):
    tables = []
    table_areas = []
    try:
        page_tables = page.find_tables()
        for table in page_tables:
            tables.append(table.extract())  
            x0, y0, x1, y1 = table.bbox
            margin_x, margin_y = 3, 2  
            table_areas.append((x0 + margin_x, y0 + margin_y, x1 - margin_x, y1 - margin_y))  
    except Exception:
        pass  
    return tables, table_areas

def extract_text_without_tables(page, table_areas):
    page_text = ""
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue  

        block_x0, block_y0, block_x1, block_y1, block_text = block[:5]
        inside_table = any(
            (bx0 <= block_x0 <= bx1 and by0 <= block_y0 <= by1) for (bx0, by0, bx1, by1) in table_areas
        )
        block_width = block_x1 - block_x0
        page_width = page.rect.width  
        is_wide_block = block_width > (page_width * 0.6)  

        if inside_table or is_wide_block:
            continue  

        page_text += block_text + " "
    
    return clean_pdf_text(page_text)

def save_chapter_to_pkl(chapter, chapter_data, output_folder="Pkl_Files/Chapters"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    chapter_filename = os.path.join(output_folder, f"chapter_{chapter}.pkl")
    with open(chapter_filename, "wb") as f:
        pickle.dump(chapter_data, f)

def load_chapter_from_pkl(chapter, input_folder="Pkl_Files/Chapters"):
    chapter_filename = os.path.join(input_folder, f"chapter_{chapter}.pkl")
    if os.path.exists(chapter_filename):
        with open(chapter_filename, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Chapter {chapter} file not found!")
        return None

def save_chunks_to_pkl(chunks, chapter, chunk_size, chunk_overlap, type):
    if(type=="text"):
        output_folder="Pkl_Files/Text_Chunks"
    else:
        output_folder="Pkl_Files/Table_Chunks"

    folder_path = os.path.join(output_folder, f"{chunk_size}_{chunk_overlap}")
    os.makedirs(folder_path, exist_ok=True)
    
    filename = os.path.join(folder_path, f"chapter_{chapter}_chunks.pkl")
    with open(filename, "wb") as f:
        pickle.dump(chunks, f)

def convert_to_text(chapter_number, tables_list):
    text_output = []
    
    for i, table in enumerate(tables_list, start=1):
        table_text = f"Chapter_{chapter_number}_Table_{i}: "
        table_rows = []
        
        for row in table:
            clean_row = [str(cell) for cell in row if cell is not None]  
            table_rows.append(" | ".join(clean_row))
        
        text_output.append("\n \n \n"+table_text+ ":\n"+ "\n".join(table_rows))
    
    return ", ".join(text_output)




#   Preprocessing.py Parent functions   #
def ProcessText(pdf_path, chapters):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        page.set_cropbox(pymupdf.Rect(20, 50, 600, 750))
    doc.save("cropped_2018_IRC_1stptg.pdf")
    pdf_path="cropped_2018_IRC_1stptg.pdf"
    
    with pymupdf.open(pdf_path) as doc:
        for chapter, (start, end) in chapters.items():
            text = ""
            tables = []
            
            for page_num in range(start - 1, end):
                page = doc[page_num]

                page_tables, table_areas = extract_tables(page)
                cleaned_text = extract_text_without_tables(page, table_areas)

                text += cleaned_text + " "
                tables.extend(page_tables)

            chapter_data = {
                "text": text.strip(),
                "tables": tables
            }

            save_chapter_to_pkl(chapter, chapter_data)

def chunk_chapters(chapters_to_chunk, type, chunk_size=1000, chunk_overlap=100):
    for chapter in chapters_to_chunk:
        chapter_data = load_chapter_from_pkl(chapter)
        if chapter_data is None:
            continue
        if(type=="text"):
            DATA = chapter_data["text"]
        else:
            DATA = convert_to_text(chapter,chapter_data["tables"])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(DATA)

        chapter_chunks = [
            Document(page_content=chunk, metadata={"chapter": chapter, "chunk_index": idx + 1})
            for idx, chunk in enumerate(chunks)
        ]
        
        save_chunks_to_pkl(chapter_chunks, chapter, chunk_size, chunk_overlap,type)




#   SaveModels.py Parent function   #
def Save_ChromaDb(Chunks,type,Model,model_name,chunk_size,chunk_overlap):
    ChromaDb_Langchain=[]
    if(type=="text"):
        Chroma_Loc="ChromaDB_Text"
    else:
        Chroma_Loc="ChromaDB_Tables"

    for chapter_chunks in Chunks:
        chapter_number = chapter_chunks[0]
        persist_dir = f"./{Chroma_Loc}/{model_name.replace('/', '-')}/{chunk_size}_{chunk_overlap}/{chapter_number}"

        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        print(f"Processing chapter: {chapter_number}")

        if not chapter_chunks[1] or all(not doc.page_content.strip() for doc in chapter_chunks[1]):
            print(f"Skipping chapter {chapter_number}: No valid content.")
            continue
        
        ChromaDb_Langchain.append(
            Chroma.from_documents(
                documents=chapter_chunks[1],
                embedding=Model,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
        )

    print(" Vector databases created.")
    return ChromaDb_Langchain

def retrieve_chunks(chunk_size, chunk_overlap,type ):
    if(type=="text"):
        input_folder="Pkl_Files/Text_Chunks"
    else:
        input_folder="Pkl_Files/Table_Chunks"

    chunk_folder = os.path.join(input_folder, f"{chunk_size}_{chunk_overlap}")
    
    if not os.path.exists(chunk_folder):
        print(f"No chunks found for size {chunk_size} and overlap {chunk_overlap} in {chunk_folder}.")
        return []

    chunk_files = [f for f in os.listdir(chunk_folder) if f.endswith(".pkl")]
    chapter_chunks_list = []

    sorted_files = sorted(chunk_files, key=lambda x: int(re.search(r"chapter_(\d+)", x).group(1)))

    for chunk_file in sorted_files:
        file_path = os.path.join(chunk_folder, chunk_file)
        with open(file_path, "rb") as f:
            chunk_data = pickle.load(f)
            
            if isinstance(chunk_data, list) and all(isinstance(doc, Document) for doc in chunk_data):
                match = re.search(r"chapter_(\d+)", chunk_file)
                chapter_name = match.group(1) if match else "unknown"

                chapter_chunks_list.append((f"chapter_{chapter_name}", chunk_data))
            else:
                print(f"Invalid data format in file: {chunk_file}")

    print(f"Retrieved chunks for {len(chapter_chunks_list)} chapters from {chunk_folder}.")
    return chapter_chunks_list

def retrieve_relevant_chunks_Pass_Chapters_To_BestChunks(persist_dir,Chunks,Model,query,top_k):
    if os.path.exists(persist_dir):
        print(f"Clearing Vector database at {persist_dir}")
        shutil.rmtree(persist_dir)
    TextQuery=Chroma.from_documents(
        documents=Chunks,
        embedding=Model,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    retriever = TextQuery.as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(query)  
    return results




#   RunModels.py helper functions   #
def extract_chapter_number(chapter_name):
    try:
        return int(chapter_name.split('_')[1])
    except (IndexError, ValueError):
        return -1

#   RunModels.py parent functions   #
def load_chroma_databases(model_name,type, chunk_size=1000, chunk_overlap=100, selected_chapters=None):
    ChromaDB_Chapters = {}
    ModelFileName = model_name.replace('/', '-')
    if(type=="text"):
        directory_path = f'./ChromaDb_Text/{ModelFileName}/{chunk_size}_{chunk_overlap}'
    else:
        directory_path = f'./ChromaDb_Tables/{ModelFileName}/{chunk_size}_{chunk_overlap}'

    Model = HuggingFaceEmbeddings(model_name=model_name) 

    files = os.listdir(directory_path)
    files_sorted = sorted(files, key=extract_chapter_number)

    for file in files_sorted:
        chapter_num = extract_chapter_number(file)
        file_path = os.path.join(directory_path, file)
        if os.path.exists(file_path) and (selected_chapters is None or chapter_num in selected_chapters):
            print(f"Loading vector database from: {file_path}")
            ChromaDB_Chapters[chapter_num] = Chroma(persist_directory=file_path, embedding_function=Model)
        else:
            print(f"Vector database doesn't exist or not selected: {file_path}")

    return [ChromaDB_Chapters[chapter] for chapter in selected_chapters if chapter in ChromaDB_Chapters]

def retrieve_relevant_chunks(query, SelectedChapters, top_k_per_chapter):
    all_results = []
    for Chapter_DB in SelectedChapters:
        retriever = Chapter_DB.as_retriever(search_kwargs={"k": top_k_per_chapter})
        results = retriever.invoke(query)  
        all_results.extend(results)  
    return all_results


#LLama model Call

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
    
    
#as_retriver from ChromaDB

# use Temp Vector Database to save and do the search again
# Try other vector databases
# No code compare 



Chapters_Intros= [
Document(metadata={'chapter': 1}, page_content='Chapter 1 Scope and Administration. This chapter contains provisions for the application, enforcement and administration of subsequent requirements of the code. In addition to establishing the scope of the code, Chapter 1 identifies which buildings and structures come under its purview. Chapter 1 is largely concerned with maintaining “due process of law” in enforcing the building criteria contained in the body of the code. Only through careful observation of the administrative provisions can the building official reasonably expect to demonstrate that “equal protection under the law” has been provided.'),
Document(metadata={'chapter': 2}, page_content='Chapter 2 Definitions. Terms defined in the code are listed alphabetically in Chapter 2. It is important to note that two chapters have their own definitions sections: Chapter 11 for the defined terms unique to energy conservation, Chapter 24 for the defined terms that are unique to fuel gas and Chapter 35 containing terms that are applicable to electrical Chapters 34 through 43. Where Chapter 24 or 35 defines a term differently than it is defined in Chapter 2, the definition applies in that chapter only. Chapter 2 definitions apply in all other locations in the code. Where understanding a term’s definition is key to or necessary for understanding a particular code provision, the term is shown in italics where it appears in the code. This is true only for those terms that have a meaning that is unique to the code. In other words, the generally understood meaning of a term or phrase might not be sufficient or consistent with the meaning prescribed by the code; therefore, it is essential that the code-defined meaning be known. Guidance regarding not only tense, gender and plurality of defined terms, but also terms not defined in this code, is provided.'),
Document(metadata={'chapter': 3}, page_content='Chapter 3 Building Planning. Chapter 3 provides guidelines for a minimum level of structural integrity, life safety, fire safety and livability for inhabitants of dwelling units regulated by this code. Chapter 3 is a compilation of the code requirements specific to the building planning sector of the design and construction process. This chapter sets forth code requirements dealing with light, ventilation, sanitation, minimum room size, ceiling height and environmental comfort. Chapter 3 establishes life-safety provisions including limitations on glazing used in hazardous areas, specifications on stairways, use of guards at elevated surfaces, window and fall protection, and rules for means of egress. Snow, wind and seismic design live and dead loads and flood-resistant construction, as well as solar energy systems, and swimming pools, spas and hot tubs, are addressed in this chapter.'),
Document(metadata={'chapter': 4}, page_content="Chapter 4 Foundations. Chapter 4 provides the requirements for the design and construction of foundation systems for buildings regulated by this code. Provisions for seismic load, flood load and frost protection are contained in this chapter. A foundation system consists of two interdependent components: the foundation structure itself and the supporting soil. The prescriptive provisions of this chapter provide requirements for constructing footings and walls for foundations of wood, masonry, concrete and precast concrete. In addition to a foundation's ability to support the required design loads, this chapter addresses several other factors that can affect foundation performance. These include controlling surface water and subsurface drainage, requiring soil tests where conditions warrant and evaluating proximity to slopes and minimum depth requirements. The chapter also provides requirements to minimize adverse effects of moisture, decay and pests in basements and crawl spaces."),
Document(metadata={'chapter': 5}, page_content='Chapter 5 Floors. Chapter 5 provides the requirements for the design and construction of floor systems that will be capable of supporting minimum required design loads. This chapter covers four different types: wood floor framing, wood floors on the ground, cold-formed steel floor framing and concrete slabs on the ground. Allowable span tables are provided that greatly simplify the determination of joist, girder and sheathing sizes for raised floor systems of wood framing and cold-formed steel framing. This chapter also contains prescriptive requirements for wood-framed exterior decks and their attachment to the main building.'),
Document(metadata={'chapter': 6}, page_content='Chapter 6 Wall Construction. Chapter 6 contains provisions that regulate the design and construction of walls. The wall construction covered in Chapter 6 consists of five different types: wood framed, cold-formed steel framed, masonry, concrete and structural insulated panel (SIP). The primary concern of this chapter is the structural integrity of wall construction and transfer of all imposed loads to the supporting structure. This chapter provides the requirements for the design and construction of wall systems that are capable of supporting the minimum design vertical loads (dead, live and snow loads) and lateral loads (wind or seismic loads). This chapter contains the prescriptive requirements for wall bracing and/or shear walls to resist the imposed lateral loads due to wind and seismic. Chapter 6 also regulates exterior windows and doors installed in walls. This chapter contains criteria for the performance of exterior windows and doors and includes provisions for testing and labeling, garage doors, wind-borne debris protection and anchorage details.'),
Document(metadata={'chapter': 7}, page_content='Chapter 7 Wall Covering. Chapter 7 contains provisions for the design and construction of interior and exterior wall coverings. This chapter establishes the various types of materials, materials standards and methods of application permitted for use as interior coverings, including interior plaster, gypsum board, ceramic tile, wood veneer paneling, hardboard paneling, wood shakes and wood shingles. Chapter 7 also contains requirements for the use of vapor retarders for moisture control in walls. Exterior wall coverings provide the weather-resistant exterior envelope that protects the building’s interior from the elements. Chapter 7 provides the requirements for wind resistance and water-resistive barrier for exterior wall coverings. This chapter prescribes the exterior wall coverings as well as the water-resistive barrier required beneath the exterior materials. Exterior wall coverings regulated by this section include aluminum, stone and masonry veneer, wood, hardboard, particleboard, wood structural panel siding, wood shakes and shingles, exterior plaster, steel, vinyl, fiber cement and exterior insulation finish systems.'),
Document(metadata={'chapter': 8}, page_content='Chapter 8 Roof-ceiling Construction. Chapter 8 regulates the design and construction of roofceiling systems. This chapter contains two roof-ceiling framing systems: wood framing and coldformed steel framing. Allowable span tables are provided to simplify the selection of rafter and ceiling joist size for wood roof framing and cold-formed steel framing. Chapter 8 also provides requirements for the application of ceiling finishes, the proper ventilation of concealed spaces in roofs (e.g., enclosed attics and rafter spaces), unvented attic assemblies and attic access.'),
Document(metadata={'chapter': 9}, page_content='Chapter 9 Roof Assemblies. Chapter 9 regulates the design and construction of roof assemblies. A roof assembly includes the roof deck, vapor retarder, substrate or thermal barrier, insulation, vapor retarder and roof covering. This chapter provides the requirement for wind resistance of roof coverings. The types of roof covering materials and installation regulated by Chapter 9 are: asphalt shingles, clay and concrete tile, metal roof shingles, mineral-surfaced roll roofing, slate and slate-type shingles, wood shakes and shingles, built-up roofs, metal roof panels, modified bitumen roofing, thermoset and thermoplastic single-ply roofing, sprayed polyurethane foam roofing, liquid applied coatings and photovoltaic shingles. Chapter 9 also provides requirements for roof drainage, flashing, above deck thermal insulation, rooftop-mounted photovoltaic systems and recovering or replacing an existing roof covering.'),
Document(metadata={'chapter': 10}, page_content='Chapter 10 Chimneys and Fireplaces. Chapter 10 contains requirements for the safe construction of masonry chimneys and fireplaces and establishes the standards for the use and installation of factory-built chimneys, fireplaces and masonry heaters. Chimneys and fireplaces constructed of masonry rely on prescriptive requirements for the details of their construction; the factory-built type relies on the listing and labeling method of approval. Chapter 10 provides the requirements for seismic reinforcing and anchorage of masonry fireplaces and chimneys.'),
Document(metadata={'chapter': 11}, page_content='Chapter 11 [RE] Energy Efficiency. The purpose of Chapter 11 [RE] is to provide minimum design requirements that will promote efficient utilization of energy in buildings. The requirements are directed toward the design of building envelopes with adequate thermal resistance and low air leakage, and toward the design and selection of mechanical, water heating, electrical and illumination systems that promote effective use of depletable energy resources. The provisions of Chapter 11 [RE] are duplicated from the International Energy Conservation Code—Residential Provisions, as applicable for buildings which fall under the scope of the IRC. For ease of use and coordination of provisions, the corresponding IECC—Residential Provisions section number is indicated following the IRC section number [e.g. N1102.1 (R402.1)].'),
Document(metadata={'chapter': 12}, page_content='Chapter 12 Mechanical Administration. Chapter 12 establishes the limits of applicability of the code and describes how the code is to be applied and enforced. A mechanical code, like any other code, is intended to be adopted as a legally enforceable document and it cannot be effective without adequate provisions for its administration and enforcement. The provisions of Chapter 12 establish the authority and duties of the code official appointed by the jurisdiction having authority and also establish the rights and privileges of the design professional, contractor and property owner. It also relates this chapter to the administrative provisions in Chapter 1.'),
Document(metadata={'chapter': 13}, page_content='Chapter 13 General Mechanical System Requirements. Chapter 13 contains broadly applicable requirements related to appliance listing and labeling, appliance location and installation, appliance and systems access, protection of structural elements and clearances to combustibles, among others.'),
Document(metadata={'chapter': 14}, page_content='Chapter 14 Heating and Cooling Equipment and Appliances. Chapter 14 is a collection of requirements for various heating and cooling appliances, dedicated to single topics by section. The common theme is that all of these types of appliances use energy in one form or another, and the improper installation of such appliances would present a hazard to the occupants of the dwellings, due to either the potential for fire or the accidental release of refrigerants. Both situations are undesirable in dwellings that are covered by this code.'),
Document(metadata={'chapter': 15}, page_content='Chapter 15 Exhaust Systems. Chapter 15 is a compilation of code requirements related to residential exhaust systems, including kitchens and bathrooms, clothes dryers and range hoods. The code regulates the materials used for constructing and installing such duct systems. Air brought into the building for ventilation, combustion or makeup purposes is protected from contamination by the provisions found in this chapter.'),
Document(metadata={'chapter': 16}, page_content='Chapter 16 Duct Systems. Chapter 16 provides requirements for the installation of ducts for supply, return and exhaust air systems. This chapter contains no information on the design of these systems from the standpoint of air movement, but is concerned with the structural integrity of the systems and the overall impact of the systems on the fire-safety performance of the building. This chapter regulates the materials and methods of construction which affect the performance of the entire air distribution system.'),
Document(metadata={'chapter': 17}, page_content="Chapter 17 Combustion Air. Complete combustion of solid and liquid fuel is essential for the proper operation of appliances, control of harmful emissions and achieving maximum fuel efficiency. If insufficient quantities of oxygen are supplied, the combustion process will be incomplete, creating dangerous byproducts and wasting energy in the form of unburned fuel (hydrocarbons). The byproducts of incomplete combustion are poisonous, corrosive and combustible, and can cause serious appliance or equipment malfunctions that pose fire or explosion hazards. The combustion air provisions in this code from previous editions have been deleted from Chapter 17 in favor of a single section that directs the user to NFPA 31 for oil-fired appliance combustion air requirements and the manufacturer's installation instructions for solid fuel-burning appliances. If fuel gas appliances are used, the provisions of Chapter 24 must be followed."),
Document(metadata={'chapter': 18}, page_content='Chapter 18 Chimneys and Vents. Chapter 18 regulates the design, construction, installation, maintenance, repair and approval of chimneys, vents and their connections to fuel-burning appliances. A properly designed chimney or vent system is needed to conduct the flue gases produced by a fuel-burning appliance to the outdoors. The provisions of this chapter are intended to minimize the hazards associated with high temperatures and potentially toxic and corrosive combustion gases. This chapter addresses factory-built and masonry chimneys, vents and venting systems used to vent oil-fired and solid fuel-burning appliances.'),
Document(metadata={'chapter': 19}, page_content='Chapter 19 Special Appliances, Equipment and Systems. Chapter 19 regulates the installation of fuel-burning appliances that are not covered in other chapters, such as ranges and ovens, sauna heaters, fuel cell power plants and hydrogen systems. Because the subjects in this chapter do not contain the volume of text necessary to warrant individual chapters, they have been combined into a single chapter. The only commonality is that the subjects use energy to perform some task or function. The intent is to provide a reasonable level of protection for the occupants of the dwelling.'),
Document(metadata={'chapter': 20}, page_content='Chapter 20 Boilers and Water Heaters. Chapter 20 regulates the installation of boilers and water heaters. Its purpose is to protect the occupants of the dwelling from the potential hazards associated with such appliances. A water heater is any appliance that heats potable water and supplies it to the plumbing hot water distribution system. A boiler either heats water or generates steam for space heating and is generally a closed system.'),
Document(metadata={'chapter': 21}, page_content='Chapter 21 Hydronic Piping. Hydronic piping includes piping, fittings and valves used in building space conditioning systems. Applications include hot water, chilled water, steam, steam condensate, brines and water/antifreeze mixtures. Chapter 21 regulates installation, alteration and repair of all hydronic piping systems to ensure the reliability, serviceability, energy efficiency and safety of such systems.'),
Document(metadata={'chapter': 22}, page_content='Chapter 22 Special Piping and Storage Systems. Chapter 22 regulates the design and installation of fuel oil storage and piping systems. The regulations include reference to construction standards for above-ground and underground storage tanks, material standards for piping systems (both above-ground and underground) and extensive requirements for the proper assembly of system piping and components. The purpose of this chapter is to prevent fires, leaks and spills involving fuel oil storage and piping systems, whether inside or outside structures and above or underground.'),
Document(metadata={'chapter': 23}, page_content='Chapter 23 Solar Thermal Energy Systems. Chapter 23 contains requirements for the construction, alteration and repair of all systems and components of solar thermal energy systems used for space heating or cooling, and domestic hot water heating or processing. The provisions of this chapter are limited to those necessary to achieve installations that are relatively hazard free. A solar thermal energy system can be designed to handle 100 percent of the energy load of a building, although this is rarely accomplished. Because solar energy is a low-intensity energy source and dependent on the weather, it is usually necessary to supplement a solar thermal energy system with traditional energy sources. As our world strives to find alternate means of producing power for the future, the requirements of this chapter will become more and more important over time.'),
Document(metadata={'chapter': 24}, page_content='Chapter 24 Fuel Gas. Chapter 24 regulates the design and installation of fuel gas distribution piping and systems, appliances, appliance venting systems and combustion air provisions. The definition of “Fuel gas” includes natural, liquefied petroleum and manufactured gases and mixtures of these gases. The purposes of this chapter are to establish the minimum acceptable level of safety and to protect life and property from the potential dangers associated with the storage, distribution and use of fuel gases and the byproducts of combustion of such fuels. This code also protects the personnel who install, maintain, service and replace the systems and appliances addressed herein. Chapter 24 is composed entirely of text extracted from the IFGC; therefore, whether using the IFGC or the IRC, the fuel gas provisions will be identical. Note that to avoid the potential for confusion and conflicting definitions, Chapter 24 has its own definition section.'),
Document(metadata={'chapter': 25}, page_content='Chapter 25 Plumbing Administration. The requirements of Chapter 25 do not supersede the administrative provisions of Chapter 1. Rather, the administrative guidelines of Chapter 25 pertain to plumbing installations that are best referenced and located within the plumbing chapters. This chapter addresses how to apply the plumbing provisions of this code to specific types or phases of construction. This chapter also outlines the responsibilities of the applicant, installer and inspector with regard to testing plumbing installations.'),
Document(metadata={'chapter': 26}, page_content='Chapter 26 General Plumbing Requirements. The content of Chapter 26 is often referred to as “miscellaneous,” rather than general plumbing requirements. This is the only chapter of the plumbing chapters of the code whose requirements do not interrelate. If a requirement cannot be located in another plumbing chapter, it should be located in this chapter. Chapter 26 contains safety requirements for the installation of plumbing systems and includes requirements for the identification of pipe, pipe fittings, traps, fixtures, materials and devices used in plumbing systems. If specific provisions do not demand that a requirement be located in another chapter, the requirement is located in this chapter.'),
Document(metadata={'chapter': 27}, page_content='Chapter 27 Plumbing Fixtures. Chapter 27 requires fixtures to be of the proper type, approved for the purpose intended and installed properly to promote usability and safe, sanitary conditions. This chapter regulates the quality of fixtures and faucets by requiring those items to comply with nationally recognized standards. Because fixtures must be properly installed so that they are usable by the occupants of the building, this chapter contains the requirements for the installation of fixtures.'),
Document(metadata={'chapter': 28}, page_content='Chapter 28 Water Heaters. Chapter 28 regulates the design, approval and installation of water heaters and related safety devices. The intent is to minimize the hazards associated with the installation and operation of water heaters. Although this chapter does not regulate the size of a water heater, it does regulate all other aspects of the water heater installation such as temperature and pressure relief valves, safety drip pans and connections. Where a water heater also supplies water for space heating, this chapter regulates the maximum water temperature supplied to the water distribution system.'),
Document(metadata={'chapter': 29}, page_content='Chapter 29 Water Supply and Distribution. This chapter regulates the supply of potable water from both public and individual sources to every fixture and outlet so that it remains potable and uncontaminated by cross connections. Chapter 29 also regulates the design of the water distribution system, which will allow fixtures to function properly. Because it is critical that the potable water supply system remain free of actual or potential sanitary hazards, this chapter has the requirements for providing backflow protection devices.'),
Document(metadata={'chapter': 30}, page_content='Chapter 30 Sanitary Drainage. The purpose of Chapter 30 is to regulate the materials, design and installation of sanitary drainage piping systems as well as the connections made to the system. The intent is to design and install sanitary drainage systems that will function reliably, are neither undersized nor oversized and are constructed from materials, fittings and connections whose quality is regulated by this section. This chapter addresses the proper use of fittings for directing the flow into and within the sanitary drain piping system. Materials and provisions necessary for servicing the drainage system are also included in this chapter.'),
Document(metadata={'chapter': 31}, page_content='Chapter 31 Vents. Venting protects the trap seal of each trap. The vents are designed to limit differential pressures at each trap to 1 inch of water column (249 Pa). Because waste flow in the drainage system creates pressure fluctuations that can negatively affect traps, the sanitary drainage system must have a properly designed venting system. Chapter 31 covers the requirements for vents and venting. All of the provisions set forth in this chapter are intended to limit the pressure differentials in the drainage system to a maximum of 1 inch of water column (249 Pa) above or below atmospheric pressure (i.e., positive or negative pressures).'),
Document(metadata={'chapter': 32}, page_content='Chapter 32 Traps. Traps prevent sewer gas from escaping from the drainage piping into the building. Water seal traps are the simplest and most reliable means of preventing sewer gas from entering the interior environment. This chapter lists prohibited trap types and specifies the minimum trap size for each type of fixture.'),
Document(metadata={'chapter': 33}, page_content='Chapter 33 Storm Drainage. Rainwater infiltration into the ground adjacent to a building can cause the interior of foundation walls to become wet. The installation of a subsoil drainage system prevents the buildup of rainwater on the exterior of the foundation walls. This chapter provides the specifications for subsoil drain piping. Where the discharge of the subsoil drain system is to a sump, this chapter also provides coverage for sump construction, pumps and discharge piping.'),
Document(metadata={'chapter': 34}, page_content='Chapter 34 General Requirements. This chapter contains broadly applicable, general and miscellaneous requirements including scope, listing and labeling, equipment locations and clearances for conductor materials and connections and conductor identification.'),
Document(metadata={'chapter': 35}, page_content='Chapter 35 Electrical Definitions. Chapter 35 is the repository of the definitions of terms used in the body of Part VIII of the code. To avoid the potential for confusion and conflicting definitions, Part VIII, Electrical, has its own definition chapter. Codes are technical documents and every word, term and punctuation mark can impact the meaning of the code text and the intended results. The code often uses terms that have a unique meaning in the code, which can differ substantially from the ordinarily understood meaning of the term as used outside of the code. The terms defined in Chapter 35 are deemed to be of prime importance in establishing the meaning and intent of the electrical code text that uses the terms. The user of the code should be familiar with and consult this chapter because the definitions are essential to the correct interpretation of the code and because the user may not be aware that a term is defined.'),
Document(metadata={'chapter': 36}, page_content='Chapter 36 Services. This chapter covers the design, sizing and installation of the building’s electrical service equipment and grounding electrode system. It includes an easy-to-use load calculation method and service conductor sizing table. The electrical service is generally the first part of the electrical system to be designed and installed.'),
Document(metadata={'chapter': 37}, page_content='Chapter 37 Branch Circuit and Feeder Requirements. Chapter 37 addresses the requirements for designing the power distribution system, which consists of feeders and branch circuits emanating from the service equipment. This chapter dictates the ratings of circuits and the allowable loads, the number and types of branch circuits required, the wire sizing for such branch circuits and feeders and the requirements for protection from overcurrent for conductors. A load calculation method specific to feeders is also included. This chapter is used to design the electrical system on the load side of the service.'),
Document(metadata={'chapter': 38}, page_content='Chapter 38 Wiring Methods. Chapter 38 specifies the allowable wiring methods, such as cable, conduit and raceway systems, and provides the installation requirements for the wiring methods. This chapter is primarily applicable to the “rough-in” phase of construction.'),
Document(metadata={'chapter': 39}, page_content='Chapter 39 Power and Lighting Distribution. This chapter mostly contains installation requirements for the wiring that serves the lighting outlets, receptacle outlets, appliances and switches located throughout the building. The required distribution and spacing of receptacle outlets and lighting outlets is prescribed in this chapter, as well as the requirements for ground-fault and arc-fault circuit-interrupter protection.'),
Document(metadata={'chapter': 40}, page_content='Chapter 40 Devices and Luminaires. This chapter focuses on the devices, including switches and receptacles, and lighting fixtures that are typically installed during the final phase of construction.'),
Document(metadata={'chapter': 41}, page_content='Chapter 41 Appliance Installation. Chapter 41 addresses the installation of appliances including HVAC appliances, water heaters, fixed space-heating equipment, dishwashers, garbage disposals, range hoods and suspended paddle fans.'),
Document(metadata={'chapter': 42}, page_content='Chapter 42 Swimming Pools. This chapter covers the electrical installation requirements for swimming pools, storable swimming pools, wading pools, decorative pools, fountains, hot tubs, spas and hydromassage bathtubs. The allowable wiring methods are specified along with the required clearances between electrical system components and pools, spas and tubs. This chapter includes the special grounding requirements related to pools, spas and tubs, and also prescribes the equipotential bonding requirements that are unique to pools, spas and tubs.'),
Document(metadata={'chapter': 43}, page_content='Chapter 43 Class 2 Remote-control, Signaling and Power-limited Circuits. This chapter covers the power supplies, wiring methods and installation requirements for the Class 2 circuits found in dwellings. Such circuits include thermostat wiring, alarm systems, security systems, automated control systems and doorbell systems.'),
Document(metadata={'chapter': 44}, page_content='Chapter 44 Referenced Standards.')]