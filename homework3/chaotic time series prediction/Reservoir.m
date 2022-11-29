% Time series prediction with reservoir network layout
clc
clear
%Load data
tic
training_set = readmatrix("training-set.csv");
test_set = readmatrix("test-set-3.csv");

%Initialize 
identity_matrix = eye(500).*0.01;
no_reservoir_neurons = 500;
no_input_neurons = 3;

time_steps = 500;
initial_weights = randn(500,no_input_neurons)*sqrt(0.002);
reservoir_weights = randn(500)*sqrt(2/500);

initial_states = zeros(500,1); % for reservoir neurons
reservoir_states = zeros(500,length(training_set)); % reservoir states for all training patterns.

%training
for j = 1:(length(training_set)-1)

x = training_set(:,j);
reservoir_states(:,j) = initial_states(:);
initial_states = tanh(reservoir_weights*initial_states + initial_weights*x);

end 

% output matrix
output_weights = training_set*reservoir_states' * (reservoir_states*reservoir_states' + identity_matrix)^(-1);

%Checking the test set
for j = 1:(length(test_set)-1)

x = test_set(:,j);
reservoir_states(:,j) = initial_states(:);
initial_states = tanh(reservoir_weights*initial_states + initial_weights*x);

end
Output = output_weights*initial_states;


%Prediction
for t = 1:time_steps
    
    initial_states = tanh(reservoir_weights * initial_states + initial_weights * Output);
    Output = output_weights*initial_states;

    components(:,t) = Output;
    
end

y_components = components(2,:);
csvwrite("prediction.csv",y_components);

toc