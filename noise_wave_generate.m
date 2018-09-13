clear;
warning('off','all')
warning

%% Parameter setting
SNR = 15 % select SNR you want to generate: 5 or 15
SNR_name = strcat('SNR',int2str(SNR)) % folder name that SNR test result will be saved.
SamplingF = 16000; % Sampling frequency
data_path = ['C:\Users\kangun\Desktop\speech_commands_v0.01.tar']; % data path that contains original dataset.
direc1 = dir([data_path '\Google_Speech_Command']);
noise_path = dir([data_path '\_background_noise_']);
testing_list = read_testing_list();
%% Noise audio read
noise_wave = cell(length(noise_path)-3,2);
for ii = 1:length(noise_path)-3
   [data_path '\_background_noise_\' noise_path(ii+3).name]
   [noise_wave{ii,1},fs] = audioread([data_path '\_background_noise_\' noise_path(ii+3).name]);
   assert(fs == SamplingF)
   % Noise fs is 16000
   noise_name = strsplit(noise_path(ii+3).name,'.');
   noise_wave{ii,2} = noise_name{1};
end
%%
for kk = 1:length(noise_path)-3%from pink noise
    test_count = 0
    for ii = 1:length(direc1)-2
        direc1_fullpath = [data_path '\Google_Speech_Command\' direc1(ii+2).name];
        direc2 = dir(direc1_fullpath);
        for jj = 1:length(direc2)-2
            %[direc1(ii+2).name '/' direc2(jj+2).name]
            direc2_name_nowav = split(direc2(jj+2).name,'.wav');
            if sum(strcmp(testing_list,[direc1(ii+2).name '/' direc2_name_nowav{1}]))
                test_count = test_count + 1;
                direc2_fullpath = [direc1_fullpath '\' direc2(jj+2).name];
                [orig_wave,fs] = audioread(direc2_fullpath);
                assert(fs==SamplingF)
                save_path = [data_path '\' SNR_name '\' noise_wave{kk,2} '\' direc1(ii+2).name '\'  direc2(jj+2).name];
                if exist([data_path '\' SNR_name '\' noise_wave{kk,2} '\' direc1(ii+2).name]) == 0
                    mkdir([data_path '\' SNR_name '\' noise_wave{kk,2} '\' direc1(ii+2).name])
                end
                output = v_addnoise(orig_wave, SamplingF, SNR,'',noise_wave{kk,1},SamplingF);
                assert(length(orig_wave)==length(output))
                audiowrite(save_path, output, SamplingF)
            end
            
            % TEST
%             max(orig_wave)
%             min(orig_wave)
%             output = audioread(save_path);
%             audiowrite(save_path, output, SamplingF);
%             max(output)
%             min(output)
%             20*log10(sum(orig_wave.^2)/length(output))
%             20*log10(sum(output.^2)/length(output))
%             orig_snr = snr(orig_wave)
%             output_snr = snr(output)
        end
    end
    test_count
end
test_count
%audiowrite([data_path '\clean\' direc1(ii+2).name '\'  direc2(jj+2).name],orig_wave,SamplingF)
function out = read_testing_list()
    data_path = ['C:\Users\kangun\Desktop\ªË¡¶ø‰∏¡_google_speech_command_dataset'];
    [data_path '\testing_list.txt']
    fileID = fopen([data_path '\testing_list.txt'],'rt');
    testing_list = fscanf(fileID,'%s');
    out = split(testing_list,'.wav');
end