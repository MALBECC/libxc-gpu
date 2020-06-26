/*
 Copyright (C) 2006-2007 M.A.L. Marques
 Copyright (C) 2016 M. Oliveira

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

int xc_func_info_get_number_cuda(const xc_func_info_type_cuda *info)
{
  return info->number;
}

int xc_func_info_get_kind_cuda(const xc_func_info_type_cuda *info)
{
  return info->kind;
}

char const *xc_func_info_get_name_cuda(const xc_func_info_type_cuda *info)
{
  return info->name;
}

int xc_func_info_get_family_cuda(const xc_func_info_type_cuda *info)
{
  return info->family;
}

int xc_func_info_get_flags_cuda(const xc_func_info_type_cuda *info)
{
  return info->flags;
}

const func_reference_type_cuda *xc_func_info_get_references_cuda(const xc_func_info_type_cuda *info, int number)
{
  assert(number >=0 && number < XC_MAX_REFERENCES);

  if (info->refs[number] == NULL) {
    return NULL;
  } else {
    return info->refs[number];
  }
}

int xc_func_info_get_n_ext_params_cuda(xc_func_info_type_cuda *info)
{
  assert(info!=NULL);

  return info->n_ext_params;
}

char const *xc_func_info_get_ext_params_description_cuda(xc_func_info_type_cuda *info, int number)
{
  assert(number >=0 && number < info->n_ext_params);

  return info->ext_params[number].description;
}

double xc_func_info_get_ext_params_default_value_cuda(xc_func_info_type_cuda *info, int number)
{
  assert(number >=0 && number < info->n_ext_params);

  return info->ext_params[number].value;
}
