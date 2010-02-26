#include "kill.hpp"
#include <stdio.h>

list<dep_t> flow_deps, output_deps, merged_deps;
map<string,task_t> taskMap;

// Forward Function Declarations
int parse_petit_output(std::ifstream &ifs);
int readNextSource(string line, string &source, std::ifstream &ifs);
int readNextDestination(string line, string source, std::ifstream &ifs);
int readNextTaskInfo(string line);
list<string> parseTaskParamSpace(string params);
map<string,string> parseSymbolicVars(string vars);
string skipToNext(std::ifstream &ifs);
bool isEOR(string line);
void store_dep(list<dep_t> &depList, dep_t dep);
void mergeLists(void);
string processDep(dep_t dep, string dep_set, bool isInversed);
void dumpDep(dep_t dep, string iv_set, bool isInversed);
bool isFakeVariable(string var);

string trimAll(string str);
string removeWS(string str);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// C++ code

string SetIntersector::itos(int num){
    stringstream out;
    out << num;
    return out.str();
}

void SetIntersector::setFD1(dep_t fd){
    f_dep = fd;
    composed_deps.clear();
    cmpsd_counter = 0;
}

void SetIntersector::setOD(dep_t od){
    o_dep = od;
}
  
void SetIntersector::setFD2(dep_t fd){
    f_dep2 = fd;
}

void SetIntersector::compose_FD2_OD(){
    string cmpsd = "cmpsd"+itos(cmpsd_counter);
    cmpsd += " := "+f_dep2.sets+" compose "+o_dep.sets+";";
    composed_deps.push_back(cmpsd);
    ++cmpsd_counter;
}

string SetIntersector::subtract(){
    stringstream ret_val;
    ret_val << "symbolic ";
    set<string>::iterator sym_itr;
    for(sym_itr=symbolic_vars.begin(); sym_itr != symbolic_vars.end(); ++sym_itr) {
        string sym_var = *sym_itr;
        if( sym_itr!=symbolic_vars.begin() )
            ret_val << ", ";
        ret_val << sym_var;
    }
    ret_val << ";" << endl;

    ret_val << "f1 := " << f_dep.sets << ";" << endl;

    list<string>::iterator cd_itr;
    for(cd_itr=composed_deps.begin(); cd_itr != composed_deps.end(); ++cd_itr) {
        string cd = *cd_itr;
        ret_val << cd <<  endl;
    }

    ret_val << "R := f1";
    for(int i=0; i<cmpsd_counter; i++){
        if( i )
            ret_val << " union ";
        else
            ret_val << " - ( ";

        ret_val << "cmpsd" << i;

        if( i == cmpsd_counter-1 )
            ret_val << " )";
    }
    ret_val << ";" << endl;
    ret_val << "R;" << endl;

    return ret_val.str();
}


string SetIntersector::inverse(string dep_set){
    stringstream ret_val;
    ret_val << "symbolic ";
    set<string>::iterator sym_itr;
    for(sym_itr=symbolic_vars.begin(); sym_itr != symbolic_vars.end(); ++sym_itr) {
        string sym_var = *sym_itr;
        if( sym_itr!=symbolic_vars.begin() )
            ret_val << ", ";
        ret_val << sym_var;
    }
    ret_val << ";" << endl;

    ret_val << "s := " << dep_set << ";" << endl;
    ret_val << "inverse s;" << endl;

    return ret_val.str();
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// C-like code begins
int main(int argc, char **argv){
    char *fName;

    if( argc < 2 ){
        cerr << "Usage: "<< argv[0] << " Pettit_Output_File" << endl;
        exit(-1);
    }

    fName = argv[1];
    ifstream ifs( fName );
    if( !ifs ){
        cerr << "File \""<< fName <<"\" does not exist" << endl;
        exit(-1);
    }

    parse_petit_output(ifs);
    return 0;
}

int parse_petit_output(ifstream &ifs){
    string line, source;

    flow_deps.clear();
    output_deps.clear();

    // First read the task name and parameter space information
    while( getline(ifs,line) ){
        if( readNextTaskInfo(line) < 0 )
            break;
    }

    // Then read the body of the file with the actual dependencies
    while( getline(ifs,line) ){

        if( readNextSource(line, source, ifs) ){ return -1; }
        line = skipToNext(ifs);
        while( 1 ){
            if( readNextDestination(line, source, ifs) ){ return -1; }
            line = skipToNext(ifs);
            if( line.empty() || line.find("#####") != string::npos ){ break; }
        }
    }

    mergeLists();

    return 0;
}

//DSSSSM(k,n,m) {k=0..BB-1,n=k+1..BB-1,m=k+1..BB-1} A:A(k, n),B:A(m, n),C:L(m, k),D:A(m, k),E:IPIV(m, k)
int readNextTaskInfo(string line){
    static bool in_task_section=false;

    if( !line.compare("TASK SECTION START") ){
        in_task_section=true;
        return 0;
    }

    if( !in_task_section )
        return 0;

    if( !line.compare("TASK SECTION END") ){
        in_task_section=false;
        return -1;
    }

    unsigned int lb_pos, rb_pos;
    lb_pos = line.find(" {");
    rb_pos = line.find("} ");
    if( lb_pos == string::npos || rb_pos == string::npos ){
        cerr << "ERROR: Malformed Task Info entry: \"" << line << "\"" << endl; 
        return -1;
    }
    string taskName = line.substr(0,lb_pos);
    string taskParamSpace = line.substr(lb_pos+1,rb_pos-lb_pos);
    string symVars = line.substr(rb_pos+2);

    task_t task;
    task.name = taskName;
    task.paramSpace = parseTaskParamSpace(taskParamSpace);
    task.symbolicVars = parseSymbolicVars(symVars);
    taskMap[taskName] = task;

    return 1;
}


//A:A(k, n)|B:A(m, n)|C:L(m, k)|D:A(m, k)|E:IPIV(m, k)
map<string,string> parseSymbolicVars(string vars){
    map<string, string> sVars;
    string var;
    unsigned int cm_pos, cl_pos;

    // Tasks IN() and OUT() will not have symbolic variables
    if( vars.empty() ) return sVars;

    cm_pos = vars.find("|");
    while( cm_pos != string::npos ){

        string var = vars.substr(0,cm_pos);
        cl_pos = var.find(":");
        if( cl_pos == string::npos ){
            cerr << "ERROR: Malformed Task symbolic variables: \"" << vars << "\"" << endl; 
            exit(-1);
        }
        string arrayName = removeWS(var.substr(cl_pos+1));
        string symbolic  = var.substr(0,cl_pos);
        sVars[arrayName] = symbolic;

        // skip the part of the string we just processed and start over again.
        vars = vars.substr(cm_pos+1);
        cm_pos = vars.find("|");
    }

    var = vars;
    cl_pos = var.find(":");
    if( cl_pos == string::npos ){
        cerr << "ERROR: Malformed Task symbolic variables: \"" << vars << "\"" << endl; 
        exit(-1);
    }
    string arrayName = removeWS(var.substr(cl_pos+1));
    string symbolic  = var.substr(0,cl_pos);
    sVars[arrayName] = symbolic;

    return sVars;
}

list<string> parseTaskParamSpace(string params){
    list<string> paramSpace;

    // Verify that the string starts with a left bracket and then get rid of it.
    unsigned int pos = params.find("{");
    if( pos == string::npos ){
        cerr << "ERROR: Malformed Task parameter space entry: \"" << params << "\"" << endl; 
        return paramSpace;
    }
    params = params.substr(pos+1);

    // Verify that the string ends with a right bracket and then get rid of it.
    pos = params.find("}");
    if( pos == string::npos ){
        cerr << "ERROR: Malformed Task parameter space entry: \"" << params << "\"" << endl; 
        return paramSpace;
    }
    params = params.substr(0,pos);

    // Split the string at the commas and put the parts into a list
    pos = params.find(",");
    while( pos != string::npos ){
        // Take the string before the comma and put it in the list
        paramSpace.push_back(params.substr(0,pos));
        // Get rid of the part of the string before the comma
        params = params.substr(pos+1);
        // Look for another comma and start all over again
        pos = params.find(",");
    }
    // Take the last part of the string (if any) and put it in the list
    if(!params.empty())
    paramSpace.push_back(params);
    
    return paramSpace;
}


int readNextSource(string line, string &source, ifstream &ifs){
    unsigned int pos;

    // Keep reading input lines until you hit one that matches the pattern
    pos = line.find("### SOURCE:");
    while( pos == string::npos ){
        if( !getline(ifs,line) ) return 0;
        pos = line.find("### SOURCE:");
    }

    pos = line.find(":");
    // pos+2 to skip the ":" and the empty space following it
    line = line.substr(pos+2);
    if( line.empty() ){
        cerr << "Empty SOURCE" << endl;
        return -1;
    }else{
        source = line;
//        cout << "New Source Found: "<< source << endl;
    }

    return 0;
}


// Sample format of flow/output dependencies
//
// --> DTSTRF
// flow    10: A(k,k)          -->  16: A(k,k)          (0)             [ M]
// {[k] -> [k,m] : 0 <= k < m < BB}
// exact dd: {[0]}
//
// --> DTSTRF
// output  10: A(k,k)          -->  17: A(k,k)          (0)             [ M]
// {[k] -> [k,m] : 0 <= k < m < BB}
// exact dd: {[0]}
//
int readNextDestination(string line, string source, ifstream &ifs){
    stringstream ss;
    string sink, type, srcLine, junk, dstLine;
    dep_t dep;

    // read the sink of this dependency
    unsigned int pos = line.find("===>");
    if( pos != string::npos ){
        pos = line.find(">");
        // pos+2 to skip the ">" and the empty space following it
        line = line.substr(pos+2);
        if( line.empty() ){
            cerr << "Empty sink" << endl;
            return -1;
        }else{
            sink = line;
        }
    }

    // read the details of this dependency
    if( !getline(ifs,line) || isEOR(line) ){ return 0; }

    dep.source = source;
    dep.sink = sink;
    ss << line;
    ss >> type >> srcLine >> dep.srcArray >> junk >> dstLine >> dep.dstArray >> dep.loop ;

    dep.type = type;
    if( dep.loop.find("[") != string::npos ){
        dep.loop.clear();
    }

    // Remove the ":" from the source and destination line.
    pos = srcLine.find(":");
    if( pos != string::npos ){
        dep.srcLine = atoi( srcLine.substr(0,pos).c_str() );
    }
    pos = dstLine.find(":");
    if( pos != string::npos ){
        dep.dstLine = atoi( dstLine.substr(0,pos).c_str() );
    }

    // read the sets of values for the index variables
    if( !getline(ifs,line) || isEOR(line) ){ return 0; }
    dep.sets = line;

    // Store this dependency into the map.
    if( !type.compare("flow") ){
        store_dep(flow_deps, dep);
    }else if( !type.compare("output") ){
        store_dep(output_deps, dep);
    }else{
        cerr << "Unknown type of dependency. ";
        cerr << "Only \"flow\" and \"output\" types are accepted" << endl;
        return -1;
    }

    return 0;
}


void store_dep(list<dep_t> &depList, dep_t dep){
    depList.push_back(dep);
    return;
}

// is this line an End Of Record
bool isEOR(string line){
    unsigned int pos = line.find("#########");
    if( pos != string::npos || line.empty() ){
        return true;
    }
    return false;
}

string skipToNext(ifstream &ifs){
    string line;
    while( 1 ){
        if( !getline(ifs,line) ) return string("");
        unsigned int pos = line.find("===>");
        if( pos != string::npos ){ break; }
        pos = line.find("#######");
        if( pos != string::npos ){ break; }
    }
    return line;
}

string trim(string in){
    int i;
    for(i=0; i<in.length(); ++i){
        if(in[i] != ' ')
            break;
    }
    return in.substr(i);
}

string trimAll(string str){
    unsigned int s,e;
    for(s=0; s<str.length(); ++s){
        if( str[s] != ' ' ) break;
    }
    for(e=s; e<str.length(); ++e){
        if( str[e] == ' ' ) break;
    }
    return str.substr(s,e-s);
}


string removeWS(string str){
    unsigned int s,e;
    string rslt;

    for(s=0; s<str.length(); ++s){
        if( str[s] != ' ' ) break;
    }
    for(e=s; e<str.length(); ++e){
        if( str[e] == ' ' ) continue;
        rslt += str[e];
    }
    return rslt;
}


list<string> stringToVarList( string str ){
    list<string> result;
    stringstream ss;

    ss << str;

    while (!ss.eof()) {
        string token;    
        getline(ss, token, ',');
        result.push_back(token);
    }

    return result;
}


string expressionToRange(string var, string condStr){
    string lb, ub, off;
    list<string> conditions;

    // split the condition string into a list of strings
    // each of which holding only a simple expression
    while( !condStr.empty() ){
        unsigned int pos = condStr.find_first_of("&|");
        if( pos != string::npos && pos ){
            conditions.push_back( condStr.substr(0,pos) );
            condStr = condStr.substr(pos+1);
        }else{
            conditions.push_back( condStr );
            break;
        }
        conditions.push_back( condStr );
    }

    // For every expression in the list, find the variable and then
    // look for a "<" or "<=" left and/or right of the variable.
    list<string>::iterator cnd_itr=conditions.begin();
    for (; cnd_itr!=conditions.end(); ++cnd_itr){
        unsigned int pos;
        string cond = *cnd_itr;
        pos = cond.find(var);
        // if we found the var, split the string in "left" and "right" around the var
        if( pos != string::npos ){
            string left = cond.substr(0,pos);
            string right = cond.substr(pos+var.length());

            // Process the "left" string looking for the lower bound "lb"
            pos = left.find_last_of("<");
            if( pos != string::npos ){
                // check if the comparison operator is "<=".  If so the expression to the left of the
                // operator is the lower bound, otherwise we need to offset the expression by "+1"
                if( (left.length() > pos+1) && (left[pos+1] == '=') ){
                    off = "";
                }else{
                    off = "+1";
                }

                // take the part of the string up to the "<" symbol. This should contain
                // the lower bound, probably preceeded by other stuff ending in "<" or "="
                string tmp = left.substr(0,pos);
                // find the last occurrence of "<" or "=" (left of the lower bound)
                unsigned int l_pos = tmp.find_last_of("<=");
                if( l_pos != string::npos ){
                    lb = trimAll(tmp.substr(l_pos+1));
                }else{ // if no such symbol, then the whole thing is the lower bound.
                    lb = trimAll(tmp);
                }

                lb = lb.append(off);
            }

            // Process the "right" string looking for the upper bound "ub"
            int op_len=1;
            // assume it's "<". If it ends up being "<=" we will overwrite it.
            off = "-1";
            pos = right.find("<=");
            if( pos != string::npos ){
                    off = "";
                    op_len = 2;
            }else{
                pos = right.find("<");
            }
            if( pos != string::npos ){
                // take the part of the string after the "<" or "<=" symbol. This should contain
                // the upper bound, probably followed by other stuff starting with "<".
                string tmp = right.substr(pos+op_len);

                // find the first occurrence of "<" (right of the upper bound)
                unsigned int r_pos = tmp.find("<");
                if( r_pos != string::npos ){
                    ub = trimAll(tmp.substr(0,r_pos));
                }else{ // if no such symbol, then the whole thing is the upper bound.
                    ub = trimAll(tmp);
                }

                ub = ub.append(off);
            }

        }
    }

    return (lb+".."+ub);
}


void dumpDep(dep_t dep, string iv_set, bool isInversed){
    if( isInversed ){
        cout << dep.sink << " " << dep.dstArray;
        cout << " <- " << dep.source << " " << dep.srcArray;
        cout << " " << iv_set << endl;
    }else{
        cout << dep.source << " " << dep.srcArray;
        cout << " -> " << dep.sink << " " << dep.dstArray;
        cout << " " << iv_set << endl;
    }
    return;
}

void dumpMap(stringstream &ss, map<string,string> sV){
    map<string,string>::iterator it;
    for(it=sV.begin(); it!=sV.end(); ++it){
        ss << (*it).first << "==" << (*it).second << "\n";
    }
    ss << endl;
}

string processDep(dep_t dep, string iv_set, bool isInversed){
    stringstream ss;
    string srcParams, dstParams, junk;
    unsigned int posLB, posRB, posCOL;
    list<string> srcFormals, dstFormals;

    // if it is an impossible dependency, do not print anything.
    if( iv_set.find("FALSE") != string::npos )
        return "";

    // Get the list of formal parameters of the source task (k,m,n,...)
    posLB = dep.source.find("(");
    posRB = dep.source.find(")");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed dependency source: \"" << dep.source << "\"" << endl; 
       return "";
    }
    string srcFrmlStr = dep.source.substr(posLB+1,posRB-posLB-1);
    if( isInversed )
        dstFormals = stringToVarList( srcFrmlStr );
    else
        srcFormals = stringToVarList( srcFrmlStr );

    // Get the list of formal parameters of the destination task (k,m,n,...)
    posLB = dep.sink.find("(");
    posRB = dep.sink.find(")");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed dependency sink: \"" << dep.sink << "\"" << endl; 
       return "";
    }
    string dstFrmlStr = dep.sink.substr(posLB+1,posRB-posLB-1);
    if( isInversed )
        srcFormals = stringToVarList( dstFrmlStr );
    else
        dstFormals = stringToVarList( dstFrmlStr );

    // Process the sets of actual parameters
    ss << iv_set;
    ss >> srcParams >> junk >> dstParams;

    // Get the list of actual parameters of the source task
    posLB = srcParams.find("[");
    posRB = srcParams.find("]");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed set: \"" << iv_set << "\"" << endl; 
       return "";
    }
    srcParams = srcParams.substr(posLB+1,posRB-posLB-1);
    list<string> srcActuals = stringToVarList(srcParams);

    // Get the list of actual parameters of the destination task
    posLB = dstParams.find("[");
    posRB = dstParams.find("]");
    if( posLB == string::npos || posRB == string::npos){
       cerr << "Malformed set: \"" << iv_set << "\"" << endl; 
       return "";
    }
    dstParams = dstParams.substr(posLB+1,posRB-posLB-1);
    list<string> dstActuals = stringToVarList(dstParams);

    // Get the conditions that Omega told us and clean up the string
    string cond = ss.str();
    posCOL = cond.find(":");
    posRB = cond.find("}");
    if( posCOL == string::npos || posRB == string::npos ){
       cerr << "Malformed conditions: \"" << iv_set << "\"" << endl; 
       return "";
    }
    cond = trim(cond.substr(posCOL+1, posRB-posCOL-1));

    // Remove the formals from the dep.source string
    posLB = dep.source.find("(");
    if( posLB == string::npos ){
       cerr << "Malformed dependency source: \"" << dep.source << "\"" << endl; 
       return "";
    }
    string source = dep.source.substr(0,posLB);

    // Remove the formals from the dep.sink string
    posLB = dep.sink.find("(");
    if( posLB == string::npos ){
       cerr << "Malformed dependency sink: \"" << dep.sink << "\"" << endl; 
       return "";
    }
    string sink = dep.sink.substr(0,posLB);
    
    

    if( srcFormals.size() != srcActuals.size() ){
        cerr << "ERROR: source formal count != source actual count" << endl;
    }

    list<string>::iterator srcF_itr;
    list<string>::iterator srcA_itr;

    if(isInversed){
        // For every source actual that is an "In_1" type variable (i.e. "In_" followed by number)
        // which is what Omega will introduce when we inverse the sets, replace it with the
        // corresponding formal, in the source and destination sets as well as in the conditions.
        srcF_itr=srcFormals.begin();
        srcA_itr=srcActuals.begin();
        for (; srcF_itr!=srcFormals.end(); ++srcF_itr, ++srcA_itr){
            string fParam = *srcF_itr;
            string aParam = *srcA_itr;
            if( isFakeVariable(aParam) && fParam.compare(aParam) != 0 ){
                // Replace the variable in the actual parameter with the one from the formal
                *srcA_itr = fParam;
                // Do the same for all the destination actuals
                list<string>::iterator dstA_itr=dstActuals.begin();
                for (; dstA_itr!=dstActuals.end(); ++dstA_itr){
                    string dstaParam = *dstA_itr;
                    unsigned int pos = dstaParam.find(aParam);
                    if( pos != string::npos ){
                        (*dstA_itr).replace(pos,aParam.length(), fParam);
                    }
                }
                // Do the same for all the occurances of the variable in the condition
                unsigned int pos = cond.find(aParam);
                while( pos != string::npos ){
                    cond.replace(pos,aParam.length(), fParam);
                    pos = cond.find(aParam);
                }
            }
        }
    }

    // For every source actual that is not the same variable as the formal
    // (i.e. it's an expression) add the condition (formal_variable=actual_expression)
    srcF_itr=srcFormals.begin();
    srcA_itr=srcActuals.begin();
    for (; srcF_itr!=srcFormals.end(); ++srcF_itr, ++srcA_itr){
        string fParam = *srcF_itr;
        string aParam = *srcA_itr;
        if( fParam.compare(aParam) != 0 ){ // if they are different
            cond = cond.append(" && ("+fParam+"="+aParam+") ");
        }
    }

    list<string> actual_parameter_list;

    // For every destination formal, check if it exists among the source formals.
    // If it doesn't and the corresponding destination actual is not an expression
    // (i.e. the actual is the same as the formal) replace it with a lb..ub expression.
    // However, if the source task is the special task "IN()", just replace the actuals
    // with the actuals of the destination array.
    list<string>::iterator dstF_itr=dstFormals.begin();
    list<string>::iterator dstA_itr=dstActuals.begin();
    for (; dstF_itr!=dstFormals.end(); ++dstF_itr, ++dstA_itr){
        bool found=false;
        string dstfParam = *dstF_itr;
        string dstaParam = *dstA_itr;
        // look through the source formals to see if it's there
        list<string>::iterator srcF_itr=srcFormals.begin();
        for (; srcF_itr!=srcFormals.end(); ++srcF_itr){
            string srcfParam = *srcF_itr;
            if( !srcfParam.compare(dstfParam) ){
                found = true;
                break;
            }
        }
        if( found ){
            actual_parameter_list.push_back(dstaParam);
            continue;
        }
        // if we didn't find the destination formal among the source formals, check if the
        // destination actual is the same as the destination formal.  If it is, convert it
        // to a range.
        if( !dstaParam.compare(dstfParam) && !isInversed ){
            string range = expressionToRange(dstaParam, cond);
            actual_parameter_list.push_back(range);
        }else{ // if formal!=actual then it's probably an expression of source actuals
            actual_parameter_list.push_back(dstaParam);
        }
    }
    if( !isInversed ){
    }

    // iterate over the newly created list of actual destination parameters and
    // create a comma separeted list in a string, so we can print it.
    string dstTaskParams;
    list<string>::iterator a_itr=actual_parameter_list.begin();
    for (; a_itr!=actual_parameter_list.end(); ++a_itr){
        string dstActualParam = *a_itr;
        if( a_itr != actual_parameter_list.begin() ){
            dstTaskParams = dstTaskParams.append(",");
        }
        dstTaskParams = dstTaskParams.append(dstActualParam);
    }


    task_t thisTask;
    task_t peerTask;

    if( isInversed ){
        thisTask = taskMap[dep.sink];
        peerTask = taskMap[dep.source];
    }else{
        thisTask = taskMap[dep.source];
        peerTask = taskMap[dep.sink];
    }

    ss.str("");
    if( isInversed ){
        map<string,string> lcl_sV = thisTask.symbolicVars;
        map<string,string> rmt_sV = peerTask.symbolicVars;

        ss << "\n  /*" << lcl_sV[dep.dstArray] << " == " << dep.dstArray << "*/\n";
        if( dep.source.find("IN") == string::npos )
            ss << "  /*" << rmt_sV[dep.srcArray] << " == " << dep.srcArray << "*/\n";

        ss << "  IN " << lcl_sV[dep.dstArray] << " <- ";
        // If it's input we don't care about the fake array assignment in the petit file,
        // rather we use the fact that the dstArray has a literal meaning and it is an
        // actual array that exists in memory.
        if( dep.source.find("IN") != string::npos ){
            ss << dep.dstArray;
        }else{
            ss << rmt_sV[dep.srcArray];
        }
        ss << " " << source << "("<< dstTaskParams <<") ";
    }else{
        map<string,string> lcl_sV = thisTask.symbolicVars;
        map<string,string> rmt_sV = peerTask.symbolicVars;

        ss << "\n  /*" << lcl_sV[dep.srcArray] << " == " << dep.srcArray << "*/\n";
        if( dep.sink.find("OUT") == string::npos )
            ss << "  /*" << rmt_sV[dep.dstArray] << " == " << dep.dstArray << "*/\n";

        ss << "  OUT " << lcl_sV[dep.srcArray] << " -> ";
        if( dep.sink.find("OUT") != string::npos )
            ss << dep.dstArray << " ";
        else
            ss << rmt_sV[dep.dstArray] << " ";
        ss << sink << "(" << dstTaskParams << ")  ";
    }
    ss << "{" << cond << "}";

    return ss.str();
}

bool isFakeVariable(string var){
    if( var.find("In_") != string::npos ) return true;
    if( !var.compare("ii") ) return true;
    if( !var.compare("jj") ) return true;

    return false;
}

void mergeLists(void){
    bool found;
    list<dep_t>::iterator fd_itr; // flow dep iterator
    list<dep_t>::iterator od_itr; // out dep iterator
    list<dep_t>::iterator fd2_itr; // second flow dep iterator
    set<int> srcSet;
    set<int>::iterator src_itr;
    set<string> fake_it;

    fake_it.insert("BB");
    fake_it.insert("step");
    fake_it.insert("NT");
    fake_it.insert("ip");
    fake_it.insert("proot");
    fake_it.insert("P");
    fake_it.insert("B");

    SetIntersector setIntersector(fake_it);

    // Insert every source of flow deps in a set (srcSet).
    for(fd_itr=flow_deps.begin(); fd_itr != flow_deps.end(); ++fd_itr) {
        found = false;
        dep_t f_dep = *fd_itr;
        srcSet.insert(f_dep.srcLine);
    }

    // For every source of a flow dep
    for (src_itr=srcSet.begin(); src_itr!=srcSet.end(); ++src_itr){
        int source = static_cast<int>(*src_itr);
        list<dep_t> rlvnt_flow_deps;
        // Find all flow deps that flow from this source
        for(fd_itr=flow_deps.begin(); fd_itr != flow_deps.end(); ++fd_itr) {
            dep_t f_dep = static_cast<dep_t>(*fd_itr);
            int fd_srcLine = f_dep.srcLine;
            int fd_dstLine = f_dep.dstLine;
            if( fd_srcLine == source ){
                rlvnt_flow_deps.push_back(f_dep);
            }
        }

        // Iterate over all the relevant flow dependencies, apply the output dependencies
        // to them and print the result.
        for(fd_itr=rlvnt_flow_deps.begin(); fd_itr != rlvnt_flow_deps.end(); ++fd_itr) {
            dep_t f_dep = static_cast<dep_t>(*fd_itr);
            int fd_srcLine = f_dep.srcLine;
            int fd_dstLine = f_dep.dstLine;
            setIntersector.setFD1(f_dep);
            // Find and print every output dep that has the same source as this flow
            // dep and its destination is either a) the same as the source, or
            // b) the source of a second flow dep which has the same destination
            // as "fd_dstLine"
            for(od_itr=output_deps.begin(); od_itr != output_deps.end(); ++od_itr) {
                dep_t o_dep = *od_itr;
                int od_srcLine = o_dep.srcLine;
                int od_dstLine = o_dep.dstLine;

                if( fd_srcLine != od_srcLine ){
                    continue;
                }

                setIntersector.setOD(o_dep);

                // case (a)
                if( od_srcLine == od_dstLine ){
                    // Yes, the same as the original fd because the od is (cyclic) on myself.
                    setIntersector.setFD2(f_dep);
                    setIntersector.compose_FD2_OD();
                }else{ // case (b)
                    for(fd2_itr=flow_deps.begin(); fd2_itr != flow_deps.end(); ++fd2_itr) {
                        dep_t f2_dep = *fd2_itr;
                        int fd2_srcLine = f2_dep.srcLine;
                        int fd2_dstLine = f2_dep.dstLine;
                        if( fd2_srcLine == od_dstLine && fd2_dstLine == fd_dstLine ){
                            setIntersector.setFD2(f2_dep);
                            setIntersector.compose_FD2_OD();
                        }
                    }
                }
            }

            // Subtract the dependencies that kill our current flow dependency, from our current
            // flow dependency and see what's left (if anything).
            string sets_for_omega = setIntersector.subtract();
            fstream filestr ("/tmp/oc_in.txt", fstream::out);
            filestr << sets_for_omega << endl;
            filestr.close();

            string omegaHome="/Users/adanalis/Desktop/Research/PLASMA_Distributed/Omega";
            FILE *pfp = popen( (omegaHome+"/omega_calc/obj/oc /tmp/oc_in.txt").c_str(), "r");
            stringstream data;
            char buffer[256];
            while (!feof(pfp)){
                if (fgets(buffer, 256, pfp) != NULL){
                    data << buffer;
                }
            }
            pclose(pfp);
            // Read the dependency that comes out of Omega
            string line;
            while( getline(data, line) ){
                if( (line.find("#") == string::npos) && !line.empty() ){
                    break;
                }
            }
            // format the dependency in JDF format and store it in the proper task in the taskMap
            string outDep = processDep(f_dep, line, false);
            if( outDep.empty() )
                continue;

            // Find the task in the map (if it exists) and add the new OUT dep to it's outDeps
            task_t task;
            map<string,task_t>::iterator it;
            it = taskMap.find(f_dep.source);
            if ( it == taskMap.end() ){
                cerr << "FATAL ERROR: Task \""<< f_dep.source <<"\" does not exist in the taskMap" << endl;
                exit(-1);
            }
            task = it->second;
            if( task.name.compare(f_dep.source) ){
                cerr << "FATAL ERROR: Task name in taskMap does not match task name in flow dependency: ";
                cerr << "\"" <<task.name << "\" != \"" << f_dep.source << "\"" << endl;
                exit(-1);
            }
            task.outDeps.push_back(outDep);
            taskMap[f_dep.source] = task;

            // If this new OUT dep does not go to the exit, invert it to get an IN dep
            if( f_dep.sink.find("OUT") == string::npos ){
                // If it was a real dependency, ask Omega to revert it
                string rev_set_for_omega = setIntersector.inverse(line);
                filestr.open("/tmp/oc_in.txt", fstream::out);
                filestr << rev_set_for_omega << endl;
                filestr.close();

                data.clear();
                pfp = popen( (omegaHome+"/omega_calc/obj/oc /tmp/oc_in.txt").c_str(), "r");
                while (!feof(pfp)){
                    if (fgets(buffer, 256, pfp) != NULL){
                        data << buffer;
                    }
                }
                pclose(pfp);
                // Read the reversed dependency
                while( getline(data, line) ){
                    if( (line.find("#") == string::npos) && !line.empty() ){
                        break;
                    }
                }
                // format the reversed dependency in JDF format and print it
                string inDep = processDep(f_dep, line, true);

                // Find the task in the map (if it exists) and add the new OUT dep to it's outDeps
                it = taskMap.find(f_dep.sink);
                if ( it == taskMap.end() ){
                    cerr << "FATAL ERROR: Task \""<< f_dep.sink <<"\" does not exist in the taskMap" << endl;
                    exit(-1);
                }
                task = it->second;
                if( task.name.compare(f_dep.sink) ){
                    cerr << "FATAL ERROR: Task name in taskMap does not match task name in flow dependency: ";
                    cerr << "\"" <<task.name << "\" != \"" << f_dep.sink << "\"" << endl;
                    exit(-1);
                }
                task.inDeps.push_back(inDep);
                taskMap[f_dep.sink] = task;
            }
        }
    }

    // Print all the tasks
    map<string,task_t>::iterator it=taskMap.begin();
    for ( ; it != taskMap.end(); it++ ){
        task_t task = (*it).second;

        if( task.name.find("IN(") != string::npos || task.name.find("OUT(") != string::npos )
            continue;

        // Print the task name and its parameter space
        if( it != taskMap.begin() )
            cout << "\n\n";
        cout << "TASK: " << task.name << "{" << "\n";

        // Print the parameter space bounds
        list<string>::iterator ps_itr = task.paramSpace.begin();
        for(; ps_itr != task.paramSpace.end(); ++ps_itr)
            cout << "  " << *ps_itr << "\n";

        // Print the OUT dependencies
        list<string>::iterator od_itr = task.outDeps.begin();
        for(; od_itr != task.outDeps.end(); ++od_itr)
            cout << *od_itr << "\n";

        cout << "\n";

        // Print the IN dependencies
        list<string>::iterator id_itr = task.inDeps.begin();
        for(; id_itr != task.inDeps.end(); ++id_itr)
            cout << *id_itr << "\n";
        
        cout << "}" << endl;
    }



    return;
}
