// debug ?
boolean debug = false;

// game properties
float start_q  = 0.0;
float start_e  = 0.0;
float start_r  = 0.0;
float a_param  = 0.1;
float g_param  = 1;      // finite horizont
float l_param  = 0.9;
float e_param  = 0.01;   // greedy action - not yet implemented


int grid_width = 25;
int grid_height = 25;
int start_state_x = 1;
int start_state_y = 1;
int goal_x = 4;
int goal_y = 20;
int action_nr  = 4;


// canvas properties
int canvas_width = 250;
int canvas_height = 250;
int box_width = 10;
int box_height = 10;
int max_policy = 10000;

class State {
  float[] q;
  float[] r;
  float[] e;
  boolean terminal_state;
  State[] next;
  int x_ptr;
  int y_ptr;
  int chosed_action;
  
  State(int grid_x,int grid_y,float start_q,float start_e,float start_r,int action_nr){
    terminal_state = false;
    chosed_action = 0;
    q = new float[action_nr];
    r = new float[action_nr];
    e = new float[action_nr];
    next = new State[action_nr];
    x_ptr = grid_x;
    y_ptr = grid_y;
    
    for(int x=0; x < q.length; x++){
      q[x] = start_q;
      r[x] = start_r;
      e[x] = start_e;
    }
  }
  
  float average_q(){
    float avg = 0;
    for(int x=0; x < q.length; x++){
      avg = avg + q[x];
    }
    return avg/action_nr;
  }
  
  int next_action_exploitative(){
    return chosed_action;
  }
  
  int next_action_explorative_and_set(){
    chosed_action = floor(random(action_nr));
    return chosed_action;
  }
  
  int next_action_exploitative_and_set(){
    boolean are_all_equals = true;
    for(int x=0; x < q.length; x++){
      if(q[x] > q[chosed_action]){
        chosed_action = x;
        are_all_equals = false;
      }
    }
    if (are_all_equals) next_action_explorative_and_set();
    return chosed_action;
  }
  
}

class Agent{
  int x_ptr;
  int y_ptr;
  int steps;
  
  Agent(int x,int y){
    x_ptr = x;
    y_ptr = y;
    steps = 0;
  }
}

State[][] grid = new State[grid_width][grid_height]; 
Agent agent;
int[] old_policy = new int[max_policy];
int[] policy = new int[max_policy];

void setup(){
  size(canvas_width,canvas_height);
  
  // Agent setup
  agent = new Agent(start_state_x,start_state_y);
  //frameRate(10);
  
  // policies setup
  for(int x=0 ; x<max_policy; x++){ 
    old_policy[x]=-1;
    policy[x]=-1;   
  }

  // actions linked as follow
  // 0 -> up, 1->right, 2->down, 3->left

  // step 0: initialize
  for(int x=0 ; x<grid_width; x++)
    for(int y=0; y<grid_height; y++)
      grid[x][y] = new State(x,y,start_q,start_e,start_r,action_nr);

  // step 1: link
  for(int x=0 ; x<grid_width; x++){
    for(int y=0; y<grid_height; y++){

      // up, y+1
      if (y+1 >= grid_height){              // hit boundaries
        grid[x][y].next[0] = grid[x][y];
        grid[x][y].r[0] = -1;
      }else{                                // ok
        grid[x][y].next[0] = grid[x][y+1];
      }
      
      // right, x+1
      if (x+1 >= grid_width){               // hit boundaries
        grid[x][y].next[1] = grid[x][y];
        grid[x][y].r[1] = -1;
      }else{                                // ok
        grid[x][y].next[1] = grid[x+1][y];
      }
      
      // down, y-1
      if (y-1 < 0){                         // hit boundaries
        grid[x][y].next[2] = grid[x][y];
        grid[x][y].r[2] = -1;
      }else{                                // ok
        grid[x][y].next[2] = grid[x][y-1];
      }
      
      // left, x-1
      if (x-1 < 0){                         // hit boundaries
        grid[x][y].next[3] = grid[x][y];
        grid[x][y].r[3] = -1;
      }else{                                // ok
        grid[x][y].next[3] = grid[x-1][y];
      } 
    }
  }
  
  // step 2: set terminal state and its reward
  grid[goal_x][goal_y].terminal_state = true;
  for(int x=0 ; x<grid_width; x++)
    for(int y=0; y<grid_height; y++)
      for(int a=0; a<action_nr; a++)
        if(grid[x][y].next[a] == grid[goal_x][goal_y]) grid[x][y].r[a] = 1.0;
  
}  

void draw(){
  
  // step 1: compute agent movement
  // (I'm using Qlearning TD-lambda)
  State current_state = grid[agent.x_ptr][agent.y_ptr];
  
  if (current_state.terminal_state){
  
    // terminal state, check for policy conversion 
    boolean convergence = true;
    for(int x=0 ; x<max_policy; x++){ 
      if (old_policy[x] != policy[x]) convergence = false;
      old_policy[x] = policy[x];
      policy[x] = -1;
    }
    
    if (convergence){
      print("CONVERGENCE!");
      exit();
    }else{
      print(""+agent.steps+",");
      
     // put 0 on each eligibility trace
     for(int x=0 ; x<grid_width; x++)
        for(int y=0; y<grid_height; y++)
           for(int a=0; a<action_nr; a++)
             grid[x][y].e[a] = start_e;
              
     // reset agent
     agent.x_ptr = start_state_x;
     agent.y_ptr = start_state_y;
     agent.steps = 0;
      
    }
    

  }else{
  
    // not in a terminal state  
    
    int current_action = current_state.next_action_exploitative();
    State next_state = current_state.next[current_action];
    float delta = ( current_state.r[current_action] + (g_param * next_state.q[next_state.next_action_exploitative_and_set()] ) - current_state.q[current_action] );
    current_state.e[current_action] ++;
    
    if (debug) println("State:   " +  current_state.x_ptr + " " + current_state.y_ptr );
    if (debug) println("Action:  " +  current_action + " ");
  
    // Update q according to each state/action eligibility trace
    for(int x=0 ; x<grid_width; x++){
      for(int y=0; y<grid_height; y++){
        for(int a=0; a<action_nr; a++){
          grid[x][y].q[a] = grid[x][y].q[a] + a_param * delta * grid[x][y].e[a];
          grid[x][y].e[a] = grid[x][y].e[a] * l_param;
          if (debug) println("Trace:   " +  grid[x][y].e[a] + " ");
        }
      }
    }
    
    // record this choice
    policy[agent.steps] = current_action;
    
    // Move the agent into the next state
    agent.x_ptr = next_state.x_ptr;
    agent.y_ptr = next_state.y_ptr;
    agent.steps  ++;
    
    if (debug) println("");

  }
  
  
  // step 2: draw the grid
  for(int x=0 ; x<grid_width; x++){
    for(int y=0; y<grid_height; y++){
      if(x == agent.x_ptr && y == agent.y_ptr){
         fill(0, 255, 107);        
      }else if(grid[x][y].terminal_state){
         fill(204, 102, 0);      
      }else{
         fill((grid[x][y].average_q()+1) * 100 , grid[x][y].e[grid[x][y].chosed_action] * 200, 107);
      }
      rect(x*box_width,y*box_height,box_width,box_height);  
    }
  }  
}

