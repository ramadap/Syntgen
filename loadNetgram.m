%% Demoscript to see Netgram visualization tool for evolution of communities over time
clear all;
close all;
addpath('Netgram_Tool');


filename ='graphnodes.txt';
fid = fopen(filename, 'r');
index = 1;
while (~feof(fid))
    for i=1:2
        data = fgetl(fid);
        data = textscan(data,'%f');
        info{index,i} = cell2mat(data);
    end
    index = index + 1;
end
fclose(fid);
vargin = {{},0.5,0.1,0};                           % Provide label information, minimum weight criterion, minimum threshold for diff between avgweight and min weight criterion, original order sequence true if 1
%clear data;

output = run_script(info,vargin);

