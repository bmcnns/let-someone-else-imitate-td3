#include <iostream>

#include "Mutator.h"
#include "Program.h"

int main() {
    Program prog = Program();
    Mutator mutator = Mutator();

    std::cout << "Code before mutation" << std::endl;
    prog.displayCode();

    std::cout << "\n";
    mutator.mutateProgram(prog);
    std::cout << "\n";

    std::cout << "Code after mutation" << std::endl;
    prog.displayCode();

    return 0;
}
