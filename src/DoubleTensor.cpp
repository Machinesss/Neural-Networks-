#include "DoubleTensor.h"

std::vector<unsigned long> DoubleTensor::getIndex(unsigned long i) {
    if (i >= size) {
        throw(std::exception("i应当小于size"));
    }
    std::vector<unsigned long> v;
    unsigned long temp = i;
    for (unsigned int n = 0; n < dim; n++) {
        v.push_back(temp / mark[n]);
        temp = temp % mark[n];
    }
    return v;
}

DoubleTensor::DoubleTensor() {
    dim = 0;
    size = 0;
    data = nullptr;
    shape = nullptr;
    mark = nullptr;
}

DoubleTensor::DoubleTensor(std::vector<unsigned long> tensorShape, double *d, double defaultValue) : dim(tensorShape.size()) {
    if (tensorShape.empty()) {
        throw(std::exception("张量维度不能为零"));
    }
    shape = new unsigned long[dim];
    size = 1;
    for (unsigned int i = 0; i < tensorShape.size(); i++) {
        shape[i] = tensorShape[i];
        size *= tensorShape[i];
    }
    data = new double[size];
    if (d != nullptr) {
        memcpy(data, d, sizeof(double) * size);
    }
    else {
        for(unsigned long i = 0; i < size; i++){
            data[i] = defaultValue;
        }
    }
    mark = new unsigned long[dim];
    mark[dim - 1] = 1;
    for (int i = (int)dim - 2; i >= 0; i--) {
        mark[i] = mark[i + 1] * shape[i + 1];
    }
}

DoubleTensor::DoubleTensor(const DoubleTensor &t):dim(t.dim) {
    data = new double[t.size];
    memcpy(data, t.data, t.size * sizeof(double));
    shape = new unsigned long[dim];
    memcpy(shape, t.shape, dim * sizeof(unsigned long));
    mark = new unsigned long[dim];
    memcpy(mark, t.mark, dim * sizeof(unsigned long));
    size = t.size;
}

DoubleTensor::~DoubleTensor() {
    clear();
}

void DoubleTensor::clear() {
    if(!isEmpty()){
        delete data;
        delete shape;
        delete mark;
        data = nullptr;
        shape = nullptr;
        mark = nullptr;
    }
}

double &DoubleTensor::operator()(std::vector<unsigned long> index) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    if (index.size() != dim) {
        throw(std::exception("输入维度与张量维度不符"));
    }
    unsigned long sum = 0;
    for (int i = 0; i < index.size(); i++) {
        if (index[i] >= shape[i]) {
            throw(std::exception("索引错误"));
        }
        sum += index[i] * mark[i];
    }
    return data[sum];
}

double &DoubleTensor::operator[](unsigned long i) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    if(i >= size){
        throw(std::exception("超出下标索引"));
    }
    return data[i];
}


DoubleTensor DoubleTensor::operator*(const DoubleTensor &a) {
    if(isEmpty() || a.isEmpty()){
        throw (std::exception("张量为空"));
    }
    std::vector<unsigned long> v;
    for (int i = 0; i < dim; i++) {
        if (this->shape[i] != a.shape[i]) {
            throw(std::exception("张量维度不同，无法相乘"));
        }
        v.push_back(this->shape[i]);
    }
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = this->data[i] * a.data[i];
    }
    return DoubleTensor(v, newData);
}

void DoubleTensor::operator*=(const DoubleTensor &a) {
    if(isEmpty() || a.isEmpty()){
        throw (std::exception("张量为空"));
    }
    for (int i = 0; i < dim; i++) {
        if (this->shape[i] != a.shape[i]) {
            throw(std::exception("张量维度不同，无法相乘"));
        }
    }
    for (unsigned long i = 0; i < size; i++) {
        data[i] *= a.data[i];
    }
}

DoubleTensor DoubleTensor::operator*(const double &n) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    std::vector<unsigned long> v(this->shape, this->shape+dim);
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = this->data[i] * n;
    }
    return DoubleTensor(v, newData);
}

void DoubleTensor::operator*=(const double &n) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    for (unsigned long i = 0; i < size; i++) {
        data[i] *= n;
    }
}

DoubleTensor DoubleTensor::operator/(const DoubleTensor &a) {
    if(isEmpty() || a.isEmpty()){
        throw (std::exception("张量为空"));
    }
    std::vector<unsigned long> v;
    for (int i = 0; i < dim; i++) {
        if (this->shape[i] != a.shape[i]) {
            throw(std::exception("张量维度不同，无法相除"));
        }
        v.push_back(this->shape[i]);
    }
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = this->data[i] / a.data[i];
    }
    return DoubleTensor(v, newData);
}

DoubleTensor DoubleTensor::operator/(const double &n) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    std::vector<unsigned long> v(this->shape, this->shape+dim);
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = this->data[i] / n;
    }
    return DoubleTensor(v, newData);
}

DoubleTensor DoubleTensor::operator+(const DoubleTensor &a) {
    if(isEmpty() || a.isEmpty()){
        throw (std::exception("张量为空"));
    }
    std::vector<unsigned long> v;
    for (int i = 0; i < dim; i++) {
        if (this->shape[i] != a.shape[i]) {
            throw(std::exception("张量维度不同，无法相加"));
        }
        v.push_back(this->shape[i]);
    }
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = this->data[i] + a.data[i];
    }
    return DoubleTensor(v, newData);
}

void DoubleTensor::operator+=(const DoubleTensor &a) {
    if(isEmpty() || a.isEmpty()){
        throw (std::exception("张量为空"));
    }
    for (int i = 0; i < dim; i++) {
        if (this->shape[i] != a.shape[i]) {
            throw(std::exception("张量维度不同，无法相加"));
        }
    }
    for (unsigned long i = 0; i < size; i++) {
        data[i] += a.data[i];
    }
}

DoubleTensor DoubleTensor::operator+(const double &n) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    std::vector<unsigned long> v(this->shape, this->shape+dim);
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = this->data[i] + n;
    }
    return DoubleTensor(v, newData);
}

void DoubleTensor::operator+=(const double &n) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    for (unsigned long i = 0; i < size; i++) {
        data[i] += data[i] + n;
    }
}

DoubleTensor DoubleTensor::operator-(const DoubleTensor &a) {
    if(isEmpty() || a.isEmpty()){
        throw (std::exception("张量为空"));
    }
    std::vector<unsigned long> v;
    for (int i = 0; i < dim; i++) {
        if (this->shape[i] != a.shape[i]) {
            throw(std::exception("张量维度不同，无法相减"));
        }
        v.push_back(this->shape[i]);
    }
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = this->data[i] - a.data[i];
    }
    return DoubleTensor(v, newData);
}

DoubleTensor DoubleTensor::operator-(const double &n) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    std::vector<unsigned long> v(this->shape, this->shape+dim);
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = this->data[i] - n;
    }
    return DoubleTensor(v, newData);
}

DoubleTensor DoubleTensor::operator-() {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    std::vector<unsigned long> v(this->shape, this->shape+dim);
    double* newData = new double[this->size];
    for (unsigned long i = 0; i < size; i++) {
        newData[i] = -this->data[i];
    }
    return DoubleTensor(v, newData);
}


DoubleTensor &DoubleTensor::operator=(const DoubleTensor &a) {
    if(&a == this){
        return *this;
    }
    if(!isEmpty()){
        clear();
    }
    data = new double[a.size];
    memcpy(data, a.data, a.size * sizeof(double));
    shape = new unsigned long[dim];
    memcpy(shape, a.shape, dim * sizeof(unsigned long));
    mark = new unsigned long[dim];
    memcpy(mark, a.mark, dim * sizeof(unsigned long));
    size = a.size;
    dim = a.dim;
    return *this;
}

DoubleTensor DoubleTensor::mm(DoubleTensor &a) {
    if(isEmpty() || a.isEmpty()){
        throw (std::exception("张量为空"));
    }
    if (this->dim != 2 || a.dim != 2) {
        throw(std::exception("mm方法仅适用于dim等于2的张量"));
    }
    std::vector<unsigned long> v;
    if (this->shape[1] != a.shape[0]) {
        throw(std::exception("矩阵维度不符，无法相乘"));
    }
    v.push_back(this->shape[0]);
    v.push_back(a.shape[1]);
    DoubleTensor newT(v);
    for (unsigned long i = 0; i < newT.size; i++) {
        std::vector<unsigned long> index = newT.getIndex(i);
        double temp = 0;
        for (unsigned long j = 0; j < a.shape[0]; j++) {
            temp += this->operator()({ index[0], j }) * a({ j, index[1] });
        }
        newT.data[i] = temp;
    }
    return newT;
}

DoubleTensor DoubleTensor::pow(double n) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    std::vector<unsigned long> v(shape, shape + dim);
    DoubleTensor newT(v);
    for (int i = 0; i < newT.size; i++) {
        newT.data[i] = std::pow(this->data[i], n);
    }
    return newT;
}

DoubleTensor &DoubleTensor::pow_(double n) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    for (int i = 0; i < size; i++) {
        data[i] = std::pow(data[i], n);
    }
    return *this;
}

DoubleTensor DoubleTensor::log() {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    std::vector<unsigned long> v(shape, shape + dim);
    DoubleTensor newT(v);
    for (int i = 0; i < newT.size; i++) {
        newT.data[i] = std::log(this->data[i]);
    }
    return newT;
}

DoubleTensor &DoubleTensor::log_() {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    for (int i = 0; i < size; i++) {
        data[i] = std::log(data[i]);
    }
    return *this;
}


DoubleTensor DoubleTensor::exp() {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    std::vector<unsigned long> v(shape, shape + dim);
    DoubleTensor newT(v);
    for (int i = 0; i < newT.size; i++) {
        newT.data[i] = std::exp(this->data[i]);
    }
    return newT;
}

DoubleTensor DoubleTensor::exp_() {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    for (int i = 0; i < size; i++) {
        data[i] = std::exp(data[i]);
    }
    return *this;
}


DoubleTensor DoubleTensor::slice(unsigned long i) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    if (i >= shape[0]) {
        throw(std::exception("输入维度与张量维度不符"));
    }
    unsigned long begin = i * mark[0];
    double* temp = &data[begin];
    double* a = new double[mark[0]];
    memcpy(a, temp, sizeof(double) * mark[0]);
    std::vector<unsigned long> v;
    for (int n = 1; n < dim; n++) {
        v.push_back(shape[n]);
    }
    return DoubleTensor(v, a);
}

DoubleTensor DoubleTensor::slice(unsigned long i, unsigned long j) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    if (i >= shape[0] || j >= shape[0]) {
        throw(std::exception("输入维度与张量维度不符"));
    }
    if (i >= j) {
        throw(std::exception("Error"));
    }
    unsigned long begin = i * mark[0];
    double* temp = &data[begin];
    double* a = new double[mark[0] * (j - i + 1)];
    memcpy(a, temp, sizeof(double) * mark[0] * (j - i + 1));
    std::vector<unsigned long> v;
    v.push_back(j - i + 1);
    for (int n = 1; n < dim; n++) {
        v.push_back(shape[n]);
    }
    return DoubleTensor(v, a);
}

void DoubleTensor::display(unsigned int temp, unsigned long begin, unsigned long end, bool flag) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    end = end == -1 ? size : end;
    if (temp == dim - 1) {
        printf("[ ");
        for (unsigned long i = begin; i < end; i++) {
            printf("%5.4lf ", data[i]);
        }
        if (!flag) {
            printf("]\n");
        }
        else {
            printf("]");
        }
    }
    else {
        printf("[");
        for (int n = 0; n < shape[temp]; n++) {
            if (n == shape[temp] - 1) {
                display(temp + 1, begin + mark[temp] * n, begin + mark[temp] * (n + 1), true);
            }
            else {
                display(temp + 1, begin + mark[temp] * n, begin + mark[temp] * (n + 1));
            }
        }
        if (!flag) {
            printf("]\n");
        }
        else {
            printf("]");
        }
    }
}

unsigned long DoubleTensor::getSize() const {
    return size;
}

const unsigned long *DoubleTensor::getShape() {
    return shape;
}

DoubleTensor DoubleTensor::expand(std::vector<unsigned long> newShape) {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    if(newShape.size() < dim){
        throw (std::exception("不允许高维向低维转换"));
    }
    else if(newShape.size() > dim){
        unsigned int times = 1;
        unsigned long newSize = 1;
        for(unsigned int i = 0, j = 0; i < newShape.size(); i++){
            if(i < newShape.size() - dim){
                times *= newShape[i];
            }
            else if(newShape[i] != shape[j++]){
                throw (std::exception("扩充后低维度应与原维度相同"));
            }
            newSize *= newShape[i];
        }
        double *newData = new double[newSize];
        for(unsigned int i = 0; i < times; i++){
            memcpy(newData + i*size, data, sizeof(double) * size);
        }
        return DoubleTensor(newShape, newData);
    }
    else{
        if(shape[1] == 1){
            for(unsigned int i = 0; i < dim - 1; i++){
                if(newShape[i] != shape[i]){
                    throw (std::exception("扩充后低维度应与原维度相同"));
                }
            }
            unsigned int times = newShape[dim - 1];
            unsigned long newSize = newShape[dim - 1] * size;
            double *newData = new double[newSize];
            for(unsigned i = 0, num = 0; i < size && num < newSize; i++){
                for(unsigned int n = 0; n < times; n++){
                    newData[num++] = data[i];
                }
            }
            return DoubleTensor(newShape, newData);
        }
        else if(shape[0] == 1){
            for(unsigned int i = 1; i < newShape.size(); i++){
                    if(newShape[i] != shape[i]){
                    throw (std::exception("扩充后低维度应与原维度相同"));
                }
            }
            unsigned int times = newShape[0];
            unsigned long newSize = newShape[0] * size;
            double *newData = new double[newSize];
            for(unsigned int i = 0; i < times; i++){
                memcpy(newData + i*size, data, sizeof(double) * size);
            }
            return DoubleTensor(newShape, newData);
        }
        else {
            throw (std::exception("扩充维度不合法"));
        }
    }
}

DoubleTensor DoubleTensor::transpose() {
    if(isEmpty()){
        throw (std::exception("该张量为空"));
    }
    if(dim > 2){
        throw (std::exception("矩阵转置仅适用于dim<=2"));
    }
    double *data = new double[size];
    unsigned long num = 0;
    for(unsigned long j = 0; j < shape[1]; j++){
        for(unsigned long i = 0; i < shape[0]; i++){
            data[num ++] = this->operator()({i, j});
        }
    }
    return DoubleTensor({shape[1], shape[0]}, data);
}

std::vector<unsigned long> DoubleTensor::getVectorShape() const{
    return {shape, shape+dim};
}

unsigned int DoubleTensor::getDim() const {
    return dim;
}

bool DoubleTensor::isEmpty() const{
    if(data == nullptr){
        return true;
    }
    return false;
}
