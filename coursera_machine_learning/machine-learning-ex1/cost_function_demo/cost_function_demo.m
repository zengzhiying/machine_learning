
X = [1 2; 1 3; 1 6];
y = [4; 8; 9];
theta0 = 0;
theta1 = 1;
theta = [theta0; theta1];
J = costFunctionJ(X, y, theta);
disp(J);