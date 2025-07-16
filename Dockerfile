FROM node:20-alpine

WORKDIR /src

COPY package*.json ./

RUN npm ci

COPY . .

EXPOSE 3010

# CMD ["npm","run","start:dev"]
CMD ["npm","run","lg:dev"]
