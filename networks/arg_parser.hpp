/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class ArgParser {
  struct Option {
    std::string name{""};
    std::string description{""};
    std::vector<std::string> supportedVals = {};
    bool parsed = false;
    bool required = true;
  };

  std::string program_name_;
  int argc_;
  char** argv_;
  std::vector<Option> opts_;
  std::unordered_map<std::string, std::string> parsed_;

 public:
  ArgParser() = delete;
  ArgParser(int argc, char** argv)
      : program_name_(argv[0]), argc_(argc), argv_(argv) {}
  ArgParser& operator=(const ArgParser& other) {
    if (this != &other) {
      program_name_ = other.program_name_;
      argc_ = other.argc_;
      argv_ = other.argv_;
      opts_ = other.opts_;
      parsed_ = other.parsed_;
    }

    return *this;
  }
  ArgParser& operator=(ArgParser&& other) {
    if (this != &other) {
      program_name_ = std::move(other.program_name_);
      argc_ = other.argc_;
      argv_ = other.argv_;
      other.argv_ = nullptr;
      opts_ = std::move(other.opts_);
      parsed_ = std::move(other.parsed_);
    }

    return *this;
  }
  ArgParser(const ArgParser& other) { *this = other; }
  ArgParser(ArgParser&& other) { *this = std::move(other); }
  ~ArgParser() = default;

  bool parseArgs() {
    for (int i = 1; i < argc_; ++i) {
      std::string optStr(argv_[i]);
      auto it = std::find_if(opts_.begin(), opts_.end(), [optStr](Option& opt) {
        return opt.name == optStr;
      });

      if (it == opts_.end()) {
        std::cerr << "Unknown option " << optStr << "\n";
        return false;
      }

      if (i + 1 >= argc_) {
        std::cerr << "No value provided for option " << optStr << "\n";
        return false;
      }

      std::string valStr(argv_[i + 1]);
      i++;
      if (valStr.find("-") == 0) {
        std::cerr << "Invalid value " << valStr << " for option " << optStr
                  << "\n";
        return false;
      }

      if (it->parsed) {
        std::cerr << "Duplicate option " << optStr << "\n";
        return false;
      }

      if (!it->supportedVals.empty()) {
        bool valid = false;
        for (const auto& supportedVal : it->supportedVals) {
          if (supportedVal == valStr) {
            valid = true;
            break;
          }
        }
        if (!valid) {
          std::cerr << "Unsupported value " << valStr << " for option "
                    << optStr << "\n";
          return false;
        }
      }

      it->parsed = true;
      parsed_.insert({optStr, valStr});
    }

    for (const auto& opt : opts_) {
      if (!opt.parsed && opt.required) {
        std::cerr << "Required argument " << opt.name << " not passed\n";
        return false;
      }
    }

    return true;
  }

  void addOptWithSupportedValues(const std::string& name,
                                 const std::string& description,
                                 const std::vector<std::string> supportedVals,
                                 bool required = true) {
    Option option;
    option.name = name;
    option.description = description;
    option.supportedVals = supportedVals;
    option.required = required;
    opts_.emplace_back(std::move(option));
  }

  void addOpt(const std::string& name, const std::string& description,
              bool required = true) {
    addOptWithSupportedValues(name, description, {}, required);
  }

  std::string getValueForOpt(const std::string& opt,
                             bool warnIfNotFound = true) const {
    auto it = parsed_.find(opt);
    if (it != parsed_.end()) {
      return it->second;
    }

    if (warnIfNotFound) {
      std::cerr << "Value for option " << opt << " not found\n";
    }

    return std::string("");
  }

  bool hasValueForOpt(const std::string& opt) const {
    return !getValueForOpt(opt, /*warn*/ false).empty();
  }

  void printHelp() const {
    std::cout << "\nUsage: " << program_name_ << "\n";
    std::cout << "Arguments:\n";
    for (const auto& opt : opts_) {
      std::cout << "\t" << opt.name << "  " << opt.description << ". ";
      if (!opt.supportedVals.empty()) {
        std::cout << "Supported values: ";
        for (const auto& val : opt.supportedVals) {
          if (val.empty()) {
            std::cout << "\"\" (empty string)"
                      << ", ";
          } else {
            std::cout << val << ", ";
          }
        }
      }
      std::cout << (opt.required ? "(Required)" : "(Optional)") << "\n";
    }
  }

  void dump() const {
    for (auto it = parsed_.cbegin(); it != parsed_.cend(); ++it) {
      std::cout << it->first << " : " << it->second << "\n";
    }
  }
};
