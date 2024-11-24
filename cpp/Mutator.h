//
// Created by bryce on 12/10/24.
//

#ifndef MUTATOR_H
#define MUTATOR_H

#include <random>
#include "Program.h"


class Mutator {
public:
    static void mutateProgram(Program& program);
    static std::pair<Program, Program> crossover(const Program &program1, const Program &program2);
private:
    static void mutateInstruction(uint32_t& instruction);
    static std::mt19937 gen;
    static std::uniform_real_distribution<> rng;
};

#endif //MUTATOR_H
