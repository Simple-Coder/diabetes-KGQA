import request from '@/utils/request'

export function doAnswer(data) {
  return request({
    url: '/service/api/answer',
    method: 'post',
    data: data
  })
}

