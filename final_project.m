%% 4420 Final Project
filenames = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", ...
             "Subject_5", "Subject_6", "Subject_7", "Subject_8"];
num_subject = 8;
% create empty cells
n_test = zeros(1,num_subject);
n_train = zeros(1,num_subject);
n_trials = zeros(1,num_subject);

for i = 1:8
    % define filepath of targeted .mat file
    filepath = sprintf('/Users/joannaq/Desktop/Spring 23/4420/final project/Signal_Processing_FC/%s.mat', filenames(i));
    % load data
    data_struct = load(filepath);
    % keep track of facts of datasets
    n_test(i) = data_struct.n_TEST;
    n_train(i) = data_struct.n_TRAIN;
    n_trials(i) = data_struct.nTrial;
    % preprocess eegdata
    preprocess(data_struct.X_EEG_TRAIN, sprintf('%s_train', filenames(i)));
    preprocess(data_struct.X_EEG_TEST, sprintf('%s_test', filenames(i)));
    % get and save train labels
    train_label = data_struct.Y_EEG_TRAIN;
    label_path = sprintf('/Users/joannaq/Desktop/Spring 23/4420/final project/preprocessed_data/bandpass_filter_only/%s_label.mat', filenames(i)); 
    save(label_path, "train_label");
end

function [] = preprocess(data, filename)
    % get eeglab
    eeglab redraw;
    % read eegdata
    EEG = pop_importdata('dataformat','array','nbchan',60,'data', data,'srate',1000, ...
                         'chanlocs', '/Users/joannaq/Desktop/Spring 23/4420/final project/Signal_Processing_FC/standard60.loc');
    % bandpass filter
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',40); % 0.5-40 bandpass filter
    % save eegdata back to .mat
    save_path = sprintf('/Users/joannaq/Desktop/Spring 23/4420/final project/preprocessed_data/bandpass_filter_only/%s.mat',filename);
    save(save_path, 'EEG', '-v7');
    eeglab redraw;
end