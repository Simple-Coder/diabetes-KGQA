import request from '@/utils/request'

export function login(data) {
  return request({
    url: '/vue-element-admin/user/login',
    method: 'post',
    data
  })
}

export function getInfo(token) {
  return request({
    url: '/vue-element-admin/user/info',
    method: 'get',
    params: {token}
  })
}

export function logout() {
  return request({
    url: '/vue-element-admin/user/logout',
    method: 'post'
  })
}

// 注册方法
export function register(data) {
  return request({
    url: '/vue-element-admin/user/register',
    headers: {
      isToken: false
    },
    method: 'post',
    data: data
  })
}

export function getAllUsers(query) {
  return request({
    url: '/vue-element-admin/user/all',
    method: 'get',
    params: query
  })
}

export function doGetOnlineList(query) {
  return request({
    url: '/user/online/get',
    method: 'get',
    params: query
  })
}
