import { faker } from '@faker-js/faker';
import { sample } from 'lodash';

// ----------------------------------------------------------------------

const PROJECT_TITLES = [
  ['目标检测','完成',['编辑','查看结果','删除项目']],
  ['语义分割','训练中',['编辑','暂停训练']],
  ['人脸识别','准备中',['编辑','开始训练','删除项目']],
  ['图片分类','准备中',['编辑','开始训练','删除项目']],
];

const PROJECTS = [...Array(4)].map((_, index) => ({
  id: faker.datatype.uuid(),
  project: PROJECT_TITLES[index][0],
  cover: `/static/mock-images/covers/cover_${index + 1}.jpg`,
  created: faker.date.past(),
  status: PROJECT_TITLES[index][1],
  completeness: faker.datatype.number({ max: 99 }),
  operation:PROJECT_TITLES[index][2],
  creator: faker.name.findName(),
  // author: {
  //   name: faker.name.findName(),
  //   avatarUrl: `/static/mock-images/avatars/avatar_${index + 1}.jpg`,
  // },
}));

export default PROJECTS;
