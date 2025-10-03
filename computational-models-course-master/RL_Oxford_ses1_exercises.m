%%  Session 1
%%
% Exercise 1
    % Define two	variables,	called	a and	b,	with	values	14 and	37.	Multiply	
    % them	together,	storing the answer	in	a	variable	called	c.

a = 14;
b = 37; 
c = a*b; c;
% Display the result of the multiplication
disp(['The result of multiplying ', num2str(a), ' and ', num2str(b), ' is: ', num2str(c)]);

%%
% Exercise 2
    % Define	a	matrix	(grid)	of	numbers,	containing	3	rows	and	4	columns,	
    % called	myMatrix.	Make	each	element	of	the	matrix	equal	the	product	
    % of	the	row	number	and	column	number	it	belongs	to.
    % Define a 3x4 matrix
matrix = [1, 2, 3, 4; 
          2, 4, 6, 8; 
          3, 6, 9, 12]; % hard-coded


% Define the size of the matrix
rows = 3;
cols = 4;

% Initialize the matrix
myMatrix = zeros(rows, cols); % Creates a 3x4 matrix of zeros

% Fill the matrix with the product of row and column indices
for r = 1:rows
    for c = 1:cols
        myMatrix(r, c) = r * c; % Assign the product of row and column indices
    end
end

% Display the matrix
disp(matrix);
disp(myMatrix)

%%
% Exercise 3
    % Use	two	(nested) for loops	to create	a	larger version	of	myMatrix,	
    % with	25	rows	and	40	columns,	using	a	maximum	of	five	lines	of	code.

  % Define the size of the matrix
rows = 25;
cols = 40;

% Initialize the matrix
myMatrix = zeros(rows, cols); % Creates a 25X40 matrix of zeros

for r = 1:rows
    for c = 1:cols
        myMatrix(r, c) = r * c; % Assign the product of row and column indices
    end
end
 
disp(myMatrix)

% alternative
myMatrix2 = zeros(25,40);   % pre-allocate
for i = 1:25
    for j = 1:40
        myMatrix2(i,j) = i * j;
    end
end
disp(myMatrix2)

% alternative
myMatrix3 = (1:25)' * (1:40); % ' transpose is needed as by default (1:25) is a horizontal vector. — it changes one vector to a column, so dimensions match (25×1 times 1×40 → 25×40).
disp(myMatrix3)

%%
% Exercise 4. 
    % Use	indexing	to	pick	out	the	element of	myMatrix	that	tells	you	what	
    % 14*37	equals. Check	whether	it	is	equal	to	the	value	of	c

    value = myMatrix (14, 37); % value
    isEqual = (value == c);% value == c s a logical comparison in MATLAB; it asks: "Is value equal to c?". Then we add it to "isEqual" variable

    % Display the results
    disp([' element (14,37)equals = ', num2str(value)]) %num2str converts the number into a string, so you can display it with disp or concatenate it with other text.
    disp(['Value of c: ', num2str(c)]);
    disp(['Is the element equal to c? ', num2str(isEqual)]) % 1 (true) % num2str converts the number 1 or 0 into a string "1" or "0"

%%
% Exercise 5. 
    % Write a single line of code that takes the 14th row of myMatrix and
    % stores it in a new vector called myFourteenTimesTable.

    myFourteenTimesTable = myMatrix (14, :); myFourteenTimesTable

    %%
% Exercise 6. 
   %  Use the plot command to plot myFourteenTimesTable, and the
   %  imagesc command to visualize myMatrix.

   % plot

plot (myFourteenTimesTable);

% or

figure;                     % opens a new figure
plot(myFourteenTimesTable); % line plot of 14×1, 14×2, ..., 14×40
xlabel('Column number');    % x-axis label
ylabel('Value');            % y-axis label
title('14 Times Table');    % title of the plot
grid on;                    % optional: adds a grid


% visuzlize myMatrix;

figure;               % opens a new figure
imagesc(myMatrix);    % shows the 25x40 multiplication table as a color image; displays the matrix as an image, where colors represent values.
colorbar;             % adds a color scale to see values
xlabel('Column');     
ylabel('Row');        
title('Visualization of myMatrix'); 

  %%
% Exercise 7. 
    % Write a MATLAB function called reverse_order.m that takes a vector
    % as its input, and returns the vector in reverse as its output. 
    % in a new file as I need to create a function.

   % Use the reverse_order function to create a variable called myReverseFourteenTimesTable
   myReverseFourteenTimesTable = reverse_order (myFourteenTimesTable);myReverseFourteenTimesTable
