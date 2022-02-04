import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot
import datetime
import copy

def data_processor(data_path, max_ast_depth_len, max_breadth_ast_len):

    # Creates corpus of ast node types from raw ast files
    corpus = []
    for category in os.listdir(data_path):
        for data in os.listdir(data_path + category):
            path = data_path + category + "/" + data
            for file in os.listdir(path):
                my_file=open(path + "/" + file,"r")
                json_data = json.loads(my_file.read())
                for start_node in json_data:
                    corpus = corpusbuilder(start_node, corpus)

    # Create ast node type to token map
    token_dict = token_dict_constructor(corpus)

    # Intialize empty dataframe
    df = pd.DataFrame(columns =["ASTs", "Type"])
    cat_switch = {
        "Authors" : 0,
        "Tasks" : 1
    }

    # For each author / GCJ question, create a depth first + breadth first representation of all files in bundle, append ast bundle to dataframe
    for category in os.listdir(data_path):
        for data in os.listdir(data_path + category):
            ast_lists = []
            path = data_path + category + "/" + data
            for file in os.listdir(path):   
                file_contents = open(path + "/" + file,"r")
                json_data = json.loads(file_contents.read())
                file_ast = []
                for start_node in json_data:
                    node_queue = []
                    file_ast = depth_first_list_constructor(start_node, file_ast)
                    file_ast = file_ast[:max_ast_depth_len]
                    file_ast, _ = breadth_first_list_constructor(start_node, node_queue, file_ast, max_breadth_ast_len, 0)
                file_ast = tokenizer(file_ast, token_dict)
                ast_lists.append(file_ast)
            data_entry = pd.DataFrame([[ast_lists, cat_switch[category]]], columns = ["ASTs", "Type"])
            df = df.append(data_entry, ignore_index=True)

    # Return the final dataframe and length of corpus to main
    return df, len(corpus)

def corpusbuilder(start_node, corpus):

    # Creates a corpus of all ast types found in ast files using a traversal of the tree
    if start_node["type"] not in corpus:
        corpus.append(start_node["type"])
    if start_node["children"]:
        for child in start_node["children"]:
            corpus = corpusbuilder(child, corpus)
    return corpus

def token_dict_constructor(corp):
    
    # Creates a dictionary to map from ast node types to integers
    counter = 1
    token_dict = {}
    for x in corp:
        token_dict[x] = counter
        counter += 1
    return token_dict
    
def tokenizer(ast_list,  token_dict):  
    # Creates a new representation of the input ast using tokens
    token_list = []
    for node in ast_list:
        token_list.append(int(token_dict[node]))
    return token_list

def depth_first_list_constructor(start_node, file_ast):

    # Creates a depth first representaion of the input ast
    file_ast.append(start_node["type"])
    if start_node["children"]:
        for child in start_node["children"]:
            file_ast = depth_first_list_constructor(child, file_ast)
    return file_ast
        
def breadth_first_list_constructor(start_node, node_queue, file_ast, max_depth, running_counter):
    
    # Creates a depth first representation of the input ast
    # Note: May cause a recursion limit error if max depth is set too high
    running_counter += 1
    if running_counter >= max_depth:
        return file_ast, node_queue

    else:
        file_ast.append(start_node["type"])
        if start_node["children"]:
            for child in start_node["children"]:
                node_queue.append(child)
        if node_queue:
            next_node = node_queue.pop(0)
            file_ast, node_queue = breadth_first_list_constructor(next_node, node_queue, file_ast, max_depth, running_counter)
        return file_ast, node_queue

def zero_padding(list, max_len):
    
    # Zero pads the input list until length is reached
    while len(list) < max_len:
        list.append(0)
    return list

class PadInput:
    def __call__(self, batch):

        # I've decided to zero pad at batch creation rather than during dataset formation to save memory space.
        # Some asts are significantly larger than average, only zero padding to that length when required significantly cuts down on input length

        # Identifies the maximum length ast in the batch 
        internal_max_len = 0
        for x in batch:
            for y in x[1]:
                if len(y) > internal_max_len:
                    internal_max_len = len(y)

        # Updates all asts within all bundles to be equal length via zero padding
        batch_index = 0
        for x in batch:
            list_index = 0
            for y in x[1]:
                batch[batch_index][1][list_index] = zero_padding(batch[batch_index][1][list_index], internal_max_len)
                list_index += 1
            batch_index += 1

        # Packages and returns zero-padded bundles to use as training batch
        labels_list = []
        ast_list = []
        for x in batch:
            labels_list.append(torch.as_tensor(x[0]))
            ast_list.append(torch.as_tensor(x[1]))
        labels_tensor = torch.stack(labels_list)
        sequences = torch.stack(ast_list)
        return labels_tensor, sequences

class CustomDataset(Dataset):

    # Dataset class for pytorch training / validation loop
    def __init__(self, asts, labels):
        self.labels = labels
        self.asts = asts

    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
            label = self.labels[idx]
            asts = self.asts[idx]
            return label, asts

class LSTM(nn.Module):

    # Deep learning architecture specification
    def __init__(self, corpus_len, embedding_dimension, lstm_out_dimension, drop_out):
        super(LSTM, self).__init__()
        self.corpus_len = corpus_len
        self.embedding_dimension = embedding_dimension
        self.lstm_out_dimension = lstm_out_dimension
        
        self.embedding = nn.Embedding(self.corpus_len+1, self.embedding_dimension)
        self.lstm = nn.LSTM(input_size = self.embedding_dimension,
                            hidden_size = self.lstm_out_dimension,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = False)
        self.drop = nn.Dropout(p=drop_out)
        self.linear = nn.Linear(self.lstm_out_dimension, 500)
        self.linear2 = nn.Linear(500, 250)
        self.linear3 = nn.Linear(250, 1)


    def forward(self, input_tensor):
        processed_tensor = self.embedding(input_tensor)
        processed_tensor = torch.mean(processed_tensor, dim=1)
        processed_tensor = self.drop(processed_tensor)
        processed_tensor, _ = self.lstm(processed_tensor)
        processed_tensor = self.drop(processed_tensor)
        processed_tensor = self.drop(torch.mean(processed_tensor, dim=1))
        processed_tensor = torch.sigmoid(processed_tensor)
        processed_tensor = self.linear(processed_tensor)
        processed_tensor = self.linear2(processed_tensor)
        processed_tensor = self.linear3(processed_tensor)
        processed_tensor = torch.sigmoid(processed_tensor)
        processed_tensor = torch.reshape(processed_tensor, (-1,))
        return processed_tensor

def train(model,
        optimizer,
        train_loader,
        valid_loader,
        device,
        num_epochs,
        out_file_path,
        exit_option_every,
        criterion = nn.BCELoss(),
        best_valid_loss = float("Inf")):
    
    # initialize running values
    eval_every = len(train_loader)
    running_loss = 0.0
    running_valid_loss = 0.0
    running_training_accuracy = 0.0
    running_valid_accuracy = 0.0
    global_step = 0
    epoch = 0
    train_loss_list = []
    valid_loss_list = []
    train_accuracy_list = []
    valid_accuracy_list = []
    global_steps_list = []


    model.train()
    while epoch < num_epochs:

        # Controls early training exit and checkpoint rollbacks
        if epoch % exit_option_every == 0 and epoch != 0 and exit_option_every != -1:
            try:
                user_input = input("Continue training? [Y]/N : ").upper()
                if user_input != "N":
                    user_input = "Y"
            except ValueError:
                user_input = "Y"

            if user_input == "N":
                final_model_info = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict()
                }

                final_metrics_info = {
                    'epoch' : epoch,
                    'train_loss_list' : train_loss_list.copy(),
                    'valid_loss_list' : valid_loss_list.copy(),
                    'train_accuracy_list' : train_accuracy_list.copy(),
                    'valid_accuracy_list' : valid_accuracy_list.copy(),
                    'global_step' : global_step,
                    'global_steps_list' : global_steps_list.copy()
                }

                torch.save(final_model_info, out_file_path + "/model_final.pt")
                torch.save(final_metrics_info, out_file_path + "/metrics_final.pt")
                return train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, best_accuracy_pair
            
            else:
                # The user wants to continue training, ask if checkpoint rollback is desiered
                try:
                    user_input = input(f"Previous best model at epoch {checkpoint_data[0]} with validation loss {best_valid_loss:.4f}. Rollback to previous best model? Y/[N] : ").upper()
                    if user_input != "Y":
                        user_input = "N"
                except ValueError:
                    user_input = "N"

                if user_input ==  "Y":
                    model.load_state_dict(model_info['model_state_dict'])
                    optimizer.load_state_dict(model_info['optimizer_state_dict'])
                    epoch, train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, running_training_accuracy, running_valid_accuracy, running_loss, running_valid_loss, global_step, global_steps_list = checkpoint_data
                else:
                    empty = ""

        # Primary training loop
        for (labels, asts_tensors) in train_loader:  
            
            # Forward and back pass through model
            labels = labels.to(device)
            asts_tensors = asts_tensors.to(device)
            output = model(asts_tensors)
            loss = criterion(output.to(torch.float), labels.to(torch.float))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Updates running metrics for loss and accuracy
            running_training_accuracy += sklearn.metrics.accuracy_score(labels.to("cpu").detach().numpy(), torch.round(output).to("cpu").detach().numpy())
            running_loss += loss.item()
            global_step += 1

            # Evaluation loop
            if global_step % eval_every == 0:
                
                # Evaluation data is presented to model
                model.eval()
                with torch.no_grad():                    
                    for (labels, asts_tensors) in valid_loader:
                        labels = labels.to(device)
                        asts_tensors = asts_tensors.to(device)
                        output = model(asts_tensors)
                        loss = criterion(output.to(torch.float), labels.to(torch.float))
                        running_valid_accuracy += sklearn.metrics.accuracy_score(labels.to("cpu").detach().numpy(), torch.round(output).to("cpu").detach().numpy())
                        running_valid_loss += loss.item()

                # Average loss and accuracy metrics are calculated
                average_train_loss = running_loss / eval_every
                average_valid_loss = running_valid_loss / len(valid_loader)
                average_train_accuracy = running_training_accuracy / eval_every
                average_valid_accuracy = running_valid_accuracy / len(valid_loader)

                # Appends loss and accuracy values to running list for later output
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                train_accuracy_list.append(average_train_accuracy)
                valid_accuracy_list.append(average_valid_accuracy)
                global_steps_list.append(global_step)

                # Resets running variables for next training iteration
                running_loss = 0.0   
                running_training_accuracy = 0.0
                running_valid_loss = 0.0
                running_valid_accuracy = 0.0
                model.train()

                # Print training progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}, Training Accuracy: {:.4f}, Validation Accuracy: {:.4f}'
                    .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                            average_train_loss, average_valid_loss, average_train_accuracy, average_valid_accuracy))
                
                # Checkpoints model if the current validation loss is lower than previous best
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    best_accuracy_pair = (average_train_accuracy, average_valid_accuracy)
                    model_info = {
                        'model_state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict' : copy.deepcopy(optimizer.state_dict())
                    }
                    metrics_info = {
                        'epoch' : epoch,
                        'train_loss_list' : train_loss_list.copy(),
                        'valid_loss_list' : valid_loss_list.copy(),
                        'train_accuracy_list' : train_accuracy_list.copy(),
                        'valid_accuracy_list' : valid_accuracy_list.copy(),
                        'global_step' : global_step,
                        'global_steps_list' : global_steps_list.copy()
                    }
                    torch.save(model_info, out_file_path + "/model_best.pt")
                    torch.save(metrics_info, out_file_path + "/metrics_best.pt")
                    checkpoint_data = (epoch+1, train_loss_list.copy(), valid_loss_list.copy(), train_accuracy_list.copy(), valid_accuracy_list.copy(), running_training_accuracy, running_valid_accuracy, running_loss, running_valid_loss, global_step, global_steps_list.copy())
        epoch += 1    

    # Saves final model parameters and metrics as training concludes
    final_model_info = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()
    }
    final_metrics_info = {
        'epoch' : epoch,
        'train_loss_list' : train_loss_list.copy(),
        'valid_loss_list' : valid_loss_list.copy(),
        'train_accuracy_list' : train_accuracy_list.copy(),
        'valid_accuracy_list' : valid_accuracy_list.copy(),
        'global_step' : global_step,
        'global_steps_list' : global_steps_list.copy()
    }
    torch.save(final_model_info, out_file_path + "/model_final.pt")
    torch.save(final_metrics_info, out_file_path + "/metrics_final.pt")
    return train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, best_accuracy_pair

def plot(train_loss_list, valid_loss_list, train_accuracy_list, valid_accracy_list, out_path, loss_graph_title, accuracy_graph_title):
    
    # Code to plot resultant training metrics
    pyplot.figure(1)
    pyplot.plot(train_loss_list)
    pyplot.plot(valid_loss_list)
    pyplot.title(loss_graph_title)
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train Loss', 'Validation Loss'], loc='best')
    pyplot.tight_layout()
    pyplot.savefig(out_path + "/loss.png")

    pyplot.figure(2)
    pyplot.plot(train_accuracy_list)
    pyplot.plot(valid_accracy_list)
    pyplot.title(accuracy_graph_title)
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train Accuracy', 'Validation Accuracy'], loc='best')
    pyplot.tight_layout()
    pyplot.savefig(out_path + "/accuracy.png")

def main(source_data_path, out_data_path, loss_graph_title, accuracy_graph_title, max_depth_ast, max_breadth_ast, train_valid_ratio, data_loading_workers, batch_size, learning_rate, epochs, drop_out, embedding_dimension, lstm_out_dimension, exit_option_every):
    
    # Create new output folder for models and figures
    date_time = datetime.datetime.now().strftime("%y_%m_%d__%H_%M_%S")
    out_data_path = out_data_path + "/" + date_time
    os.mkdir(out_data_path)

    # Identify if cuda is available for use on system
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    data_path = source_data_path

    # Load and process data from raw .ast files, format into depth first + breadth first token lists
    print("Loading Data...")
    df, corpus_len = data_processor(data_path + "/", max_depth_ast, max_breadth_ast)
    print("Data Loaded...")
    df_1 = df[df['Type'] == 0]
    df_2 = df[df['Type'] == 1]
    df_1_train, df_1_valid = train_test_split(df_1, train_size = train_valid_ratio)
    df_2_train, df_2_valid = train_test_split(df_2, train_size = train_valid_ratio)
    df_train = pd.concat([df_1_train, df_2_train], ignore_index=True, sort=False)
    df_valid = pd.concat([df_1_valid, df_2_valid], ignore_index=True, sort=False)

    # Create data loaders for training and validation
    print("Creating Datasets...")
    train_dataset = CustomDataset(df_train["ASTs"], df_train["Type"].values.astype('float64'))
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=data_loading_workers, collate_fn=PadInput())
    valid_dataset = CustomDataset(df_valid["ASTs"], df_valid["Type"].values.astype('float64'))
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=data_loading_workers, collate_fn=PadInput())
    print("Datasets Created...")

    # Intialize deep learning model and optimizer
    print("Intializing Model...")
    model = LSTM(corpus_len=corpus_len, drop_out=drop_out, embedding_dimension=embedding_dimension, lstm_out_dimension=lstm_out_dimension).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Model Intiated...")

    # Recieve output from model training
    print("Beginning Training...")
    train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, best_accuracy_pair = train(out_file_path = out_data_path, model=model, optimizer=optimizer, num_epochs=epochs, train_loader=train_data_loader, valid_loader=valid_data_loader, device=device, exit_option_every=exit_option_every)

    # Plot metrics from training epochs
    plot(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, out_data_path, loss_graph_title, accuracy_graph_title)
    print("Training Complete")

    # Print accuracy scores from model with lowest validation error
    print(best_accuracy_pair)


if __name__ == "__main__":
    main(source_data_path = "GCJ_Arbitrary/100_25",
    out_data_path = "Output",
    loss_graph_title = 'Training Epoch vs Loss Score: 100 Authors, 100 GCJ Questions',
    accuracy_graph_title = 'Training Epoch vs Accuracy Score: 100 Authors, 100 GCJ Questions',
    max_depth_ast = 1000,
    max_breadth_ast = 500, # Note: if set to high value (>950) may cause python recursion error
    train_valid_ratio = 0.8,
    data_loading_workers = 4, # Set to 0 if you do not want multi-threaded data loading
    batch_size = 4,
    learning_rate = 0.0005,
    epochs = 200,
    drop_out = 0.15,
    embedding_dimension = 600,
    lstm_out_dimension = 1000,
    exit_option_every = 25 # Prompts the user every number of X epochs if they wish to exit training early or rollback to previous best model. Set to -1 if you do not want exit / checkpoint options
    )