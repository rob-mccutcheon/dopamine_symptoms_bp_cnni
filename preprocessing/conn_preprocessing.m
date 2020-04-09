%% Project Details
% This is pretty much the default conn processing pipeline, I have added a
% striatal matroyshka as a confound to regressout effect of cortical
% tissues near striatum

project_dir = '/home/DOPA_symptoms/';
rs_dir = fullfile(project_dir, '/data/resting_state_batch/resting_state');
project_name = fullfile(project_dir, '/results/conn_analysis_batch/conn_dopasymptoms.mat');
atlas_files = fullfile(project_dir, '/data/rois/Gordon_MNI_222a.nii')
matroyshka_files = fullfile(project_dir, '/results/striatal_matroyshka/striatal_matroyshka.nii')
rs_files = 'restingstate.nii';
struc_files = 'structural_co.nii';
atlas_name = 'gordon';
striatal_matroyshka = 'striatal_matroyshka'
NSUBJECTS=52;
TR=2;

%% Find functional and anatomical files
cd(rs_dir)
FUNCTIONAL_FILE=cellstr(conn_dir(rs_files));
STRUCTURAL_FILE=cellstr(conn_dir(struc_files));
if rem(length(FUNCTIONAL_FILE),NSUBJECTS),error('mismatch number of functional files');end
if rem(length(STRUCTURAL_FILE),NSUBJECTS),error('mismatch number of anatomical files');end
nsessions=length(FUNCTIONAL_FILE)/NSUBJECTS;
FUNCTIONAL_FILE=reshape(FUNCTIONAL_FILE,[NSUBJECTS,nsessions]);
STRUCTURAL_FILE={STRUCTURAL_FILE{1:NSUBJECTS}};
disp([num2str(size(FUNCTIONAL_FILE,1)),' subjects']);
disp([num2str(size(FUNCTIONAL_FILE,2)), ' sessions']);

%% Setup
clear BATCH;
batch.filename = project_name;
batch.parallel.N=NSUBJECTS; %parrallel processing
batch.Setup.isnew=1;
batch.Setup.nsubjects=NSUBJECTS;
batch.Setup.RT=TR;
batch.Setup.structurals=STRUCTURAL_FILE;
batch.Setup.functionals=repmat({{}},[NSUBJECTS,1]);       % Point to functional volumes for each subject/session
for nsub=1:NSUBJECTS,
    for nses=1:nsessions,
        batch.Setup.functionals{nsub}{nses}{1}=FUNCTIONAL_FILE{nsub,nses};
    end
end
%note: each subject's data is defined by three sessions and one single (4d) file per session
batch.Setup.rois.names{1}=atlas_name
batch.Setup.rois.files{1}=atlas_files;
batch.Setup.rois.multiplelabels(1)=1 %lets conn know this is an atlas multiple roi file
%batch.Setup.rois.names{2}=striatal_matroyshka
%batch.Setup.rois.files{2}=matroyshka_files;
%batch.Setup.rois.multiplelabels(2)=0


%% Preprocessing
batch.Setup.preprocessing.steps={'functional_center', 'functional_slicetime',...
                                 'functional_realign&unwarp', 'structural_center',...
                                 'structural_segment&normalize', 'functional_segment&normalize_direct',...
                                 'functional_art','functional_smooth'};
batch.Setup.preprocessing.sliceorder='interleaved (bottom-up)';
batch.Setup.preprocessing.fwhm=8;
batch.Setup.preprocessing.art_thresholds(1:2)=[5, 0.9] %these are the default ART values - 97th percentile
batch.Setup.done=1;
batch.Setup.overwrite='Yes';

%% Denoising
batch.Denoising.filter=[0.01, 0.1];
batch.Denoising.done=1;
batch.Denoising.overwrite='Yes';
%batch.Denoising.confounds.names = {striatal_matroyshka};

%% 1st Level Analysis
batch.Analysis.done=1;
batch.Analysis.overwrite='Yes';

%% Run batch
conn_batch(batch)
