clear; close; close; clc;
addpath 'C:\Users\RNG_KEYSIGHT\Desktop\Bin\Code';
addpath 'C:\Users\RNG_KEYSIGHT\Downloads\KeysightUXR-Matlab-Helper-main\KeysightUXR-Matlab-Helper-main\';
%% Experiment Information

email = 'ac286@rice.edu';
% Amount of data to collect
% num_waveforms = 20;
zero = [0 72.3];
right_edge = 90.67;

%% VISA setup

scope = scope_interface;
scope.initialize('TCPIP0::KEYSIGH-KSU9HNE.lan::inst0::INSTR');

%% Set up stages

xstart = 50;
xstep = 0.1;
xend = 150;
zstart = 256.5;
zstep = -2;
zend = 256.5;

% % Initialize Stages
xstage = initThorlabs(45446724,false);
zstage = initThorlabs(45446974, false);

% Set start and end position for X/Z
xvec = xstart:xstep:xend;
zvec = zstart:zstep:zend;

% move stages to start
xstage.SetAbsMovePos(0, xstart);
xstage.MoveAbsolute(0, xstart);
zstage.SetAbsMovePos(0, zstart);
zstage.MoveAbsolute(0, zstart);
disp("Moving...");
pause(2);
disp("Moved!");

%% Set Data Format

scope.set_data_format('WORD');
[xInc, xOrg, xRef] = scope.get_x_axis('CHAN1');
mag = scope.get_waveform('CHAN1');
num_points_td = length(mag);
xax_td = (0:num_points_td-1) * xInc + xOrg - xRef * xInc;

clear xInc xOrg xRef mag

[xInc, xOrg, xRef] = scope.get_x_axis('FUNC1');
mag = scope.get_waveform('FUNC1');
num_points_fft = length(mag);
xax_fft = (0:num_points_fft-1) * xInc + xOrg - xRef * xInc;

clear xInc xOrg xRef mag
%% FFT

ffts = zeros([length(zvec), length(xvec), num_points_fft]);
tds = zeros([length(zvec), length(xvec), num_points_td]);
scope.set_data_format('WORD');

for zindex=1:1:length(zvec)
    % email_notification(email,"Experiment",sprintf("Working on z=%d", zvec(zindex)));

    zstage.SetAbsMovePos(0, zvec(zindex));
    zstage.MoveAbsolute(0, zvec(zindex));
    pause(1);

    t1 = tic();

    for xindex=1:1:length(xvec)

        xstage.SetAbsMovePos(0, xvec(xindex));
        xstage.MoveAbsolute(0, xvec(xindex));
        pause(1);

        disp(strcat("Now: [X, Z] = ",num2str(xvec(xindex)),", ",num2str(zvec(zindex))));

        tds(zindex, xindex, :) = scope.get_waveform('CHAN1');
        ffts(zindex, xindex, :) = scope.get_waveform('FUNC1');

    end
    
    save("\Users\RNG_KEYSIGHT\Desktop\Spindel\Data\temp.mat", '-v7.3');

    t2 = toc(t1);
    disp(strcat("It takes: ", num2str(t2), " seconds."));

    xstage.SetAbsMovePos(0, xstart);
    xstage.MoveAbsolute(0, xstart);
    
    if(rem(zvec, 10))
        email_notification(email,"Experiment Update",sprintf("Checkpoint: z=%d", zvec(zindex)));
    end
end


%% Draw heatmap

% % fois = [1e9 ];
% fois = linspace(1e9, 10e9, 10);
% idxs = arrayfun(@(x) find(xax_fft > x, 1), fois);
% markers = ffts(:,:,idxs);
% 
% figure;
% % imagesc(markers(:,:,1), xvec, markers');
% surf(xvec, zvec', markers(:,:,10), 'EdgeColor', 'none'); view(2);
% colormap hot

%%
date = string(datetime('now','TimeZone','local','Format','d-MMM-y-HH'));
save(strcat('Data\',date,'.mat'),'xvec','zvec', 'tds', 'zero',  'xax_td', 'right_edge', 'ffts', 'xax_fft', '-v7.3');

%% Set stages to home

% Don't do this in a reflection experiment since it will ruin the reflector
% xstage.SetAbsMovePos(0, 0);
% xstage.MoveAbsolute(0, 0);

% zstage.SetAbsMovePos(0, 0);
% zstage.MoveAbsolute(0, 0);

%% close
scope.close()
clear scope;
email_notification(email,"Experiment","Completed");
disp("Experiment finished and data saved.");