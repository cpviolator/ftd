#pragma once 

#include <iostream>
#include <ggp_internal.h>
#include <ggp.h>
#include <comm_key.h>

namespace ggp
{

  class VoltArray {

  private:
    /**
       @brief Create the array as specified by the param
       @param[in] Parameter struct
    */
    void create(const VoltArrayParam &param);
    
    /**
       @brief Move the contents of an array to this
       @param[in,out] other Array we are moving from
    */
    void move(VoltArray &&other);
      
    uint64_t n_payload;
    uint64_t n_antennae_per_payload;
    uint64_t n_channels_per_payload;
    uint64_t n_time_per_payload;
    uint64_t n_time_inner;
    uint64_t n_polarizations;    
    
    // Pointer to the data
    quda::quda_ptr data_ptr = {};

    // Location of the array
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;

    // Storage precision of the array
    QudaPrecision storage_prec;

    // Compute precision of the array
    QudaPrecision compute_prec;

    // Create new memory or reference exisiting memory
    QudaFieldCreate create_type;
    
  public:
      
    /**
       @brief Default constructor
    */
    VoltArray() = default;
      
    /**
       @brief Copy constructor for creating a VoltArray from another VoltArray
       @param[in] field Instance of VoltArray from which we are cloning
    */
    VoltArray(const VoltArray &array) noexcept;
      
    /**
       @brief Move constructor for creating a VoltArray from another VoltArray
       @param[in] field Instance of VoltArray from which we are moving
    */
    VoltArray(VoltArray &&array) noexcept;

    /**
       @brief Constructor for creating a VoltArray from a VoltArrayParam
       @param param Contains the metadata for creating the field
    */
    VoltArray(const VoltArrayParam &param);

    /**
       @brief Destroy the array
    */
    void destroy();
    
    ~VoltArray();
    
    static VoltArray *Create(const VoltArrayParam &param) { return new VoltArray(param); }
    
    void *data() { return data_ptr.data(); }
  };
  
  class VisArray {
    
  private:
    /**
       @brief Create the array as specified by the param
       @param[in] Parameter struct
    */
    void create(const VisArrayParam &param);
    
    /**
       @brief Move the contents of an array to this
       @param[in,out] other Array we are moving from
    */
    void move(VisArray &&other);
      
    uint64_t n_base;
    uint64_t n_channels;
    uint64_t n_time;
    uint64_t n_time_inner;
    uint64_t n_polarizations;

    // Pointer to the data
    quda::quda_ptr data_ptr = {};

    // Location of the array
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;

    // Storage precision of the array
    QudaPrecision storage_prec;

    // Compute precision of the array
    QudaPrecision compute_prec;

    // Create new memory or reference exisiting memory
    QudaFieldCreate create_type;
    
  public:
      
    /**
       @brief Default constructor
    */
    VisArray() = default;
      
    /**
       @brief Copy constructor for creating a VisArray from another VisArray
       @param[in] field Instance of VisArray from which we are cloning
    */
    VisArray(const VisArray &array) noexcept;
      
    /**
       @brief Move constructor for creating a VisArray from another VisArray
       @param[in] field Instance of VisArray from which we are moving
    */
    VisArray(VisArray &&array) noexcept;

    /**
       @brief Constructor for creating a VisArray from a VisArrayParam
       @param param Contains the metadata for creating the field
    */
    VisArray(const VisArrayParam &param);

    /**
       @brief Destroy the array
    */
    void destroy();
    
    ~VisArray();
    
    static VisArray *Create(const VisArrayParam &param) { return new VisArray(param); }
    
    void *data() { return data_ptr.data(); }
  };

  class FilterBank {    
  private:
    /**
       @brief Create the array as specified by the param
       @param[in] Parameter struct
    */
    void create(const FilterBankParam &param);
    
    /**
       @brief Move the contents of an array to this
       @param[in,out] other Array we are moving from
    */
    void move(FilterBank &&other);
      
    uint64_t n_beam;
    uint64_t n_channels;
    uint64_t n_time;

    // Pointer to the data
    quda::quda_ptr data_ptr = {};

    // Location of the array
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;

    // Storage precision of the array
    QudaPrecision storage_prec;

    // Compute precision of the array
    QudaPrecision compute_prec;

    // Create new memory or reference exisiting memory
    QudaFieldCreate create_type;
    
  public:
      
    /**
       @brief Default constructor
    */
    FilterBank() = default;
      
    /**
       @brief Copy constructor for creating a FilterBank from another FilterBank
       @param[in] field Instance of FilterBank from which we are cloning
    */
    FilterBank(const FilterBank &array) noexcept;
      
    /**
       @brief Move constructor for creating a FilterBank from another FilterBank
       @param[in] field Instance of FilterBank from which we are moving
    */
    FilterBank(FilterBank &&array) noexcept;

    /**
       @brief Constructor for creating a FilterBank from a FilterBankParam
       @param param Contains the metadata for creating the field
    */
    FilterBank(const FilterBankParam &param);

    /**
       @brief Destroy the array
    */
    void destroy();
    
    ~FilterBank();
    
    static FilterBank *Create(const FilterBankParam &param) { return new FilterBank(param); }
    
    void *data() { return data_ptr.data(); }
  };

  class PipelineArrays {
    
  private:
    bool init_volt_array = false;
    bool init_vis_array = false;
    bool init_filter_bank = false;

    VoltArray *volt_array = nullptr;
    VisArray *vis_array = nullptr;
    FilterBank *filter_bank = nullptr;

    void create_volt_array(VoltArrayParam &param);
    void create_vis_array(VisArrayParam &param);
    void create_filter_bank(FilterBankParam &param);
    
    friend class XEngine;
    friend class VisibilityBeamformer;
    friend class VoltageBeamformer;
    friend class Dedispersion;

  public:
    void destroy();
    ~PipelineArrays();    
  };
  
} // namespace quda

